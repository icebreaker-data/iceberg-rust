use std::{collections::HashMap, fs::File, sync::Arc};

use datafusion::{
    arrow::{
        array::{Int32Array, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema as ArrowSchema},
        record_batch::RecordBatch,
    },
    assert_batches_eq,
    parquet::{
        arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY},
        file::properties::WriterProperties,
    },
    prelude::SessionContext,
};
use datafusion_iceberg::catalog::catalog::IcebergCatalog;
use futures::TryStreamExt;
use iceberg_rust::{
    catalog::{identifier::Identifier, tabular::Tabular, Catalog},
    object_store::ObjectStoreBuilder,
    spec::{
        manifest::{Content, DataFile, FileFormat, Status},
        namespace::Namespace,
        partition::{PartitionField, PartitionSpec, Transform},
        schema::Schema,
        types::{PrimitiveType, StructField, Type},
        values::Struct,
    },
    table::Table,
};
use iceberg_sql_catalog::SqlCatalog;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

#[cfg(feature = "proto")]
mod shipping;

const FILE_PATH_FIELD_ID: i32 = i32::MAX - 101;
const POS_FIELD_ID: i32 = i32::MAX - 102;

async fn run_query(query: &str, ctx: &SessionContext) -> Vec<RecordBatch> {
    ctx.sql(query)
        .await
        .expect("query planning failed")
        .collect()
        .await
        .expect("query execution failed")
}

/// The scan under `IcebergDvExec` must still emit the Parquet row-number
/// column after a datafusion-proto round trip, with true file positions:
/// `IcebergDvExec` deletes by them.
#[cfg(feature = "proto")]
async fn row_number_column_survives_shipping(query: &str, ctx: &SessionContext, location: &str) {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::physical_plan::collect;

    let plan = ctx
        .sql(query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut scan = None;
    plan.apply(|node| {
        if node.name() == "IcebergDvExec" {
            scan = Some(node.children()[0].clone());
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    let scan = scan.expect("position deletes are applied by IcebergDvExec");
    // The row-number column is the scan's only field with an Arrow extension type.
    let row_number = scan
        .schema()
        .fields()
        .iter()
        .find(|field| field.metadata().contains_key("ARROW:extension:name"))
        .expect("the scan emits the row-number column")
        .clone();

    let executor = shipping::executor(location);
    let shipped = shipping::ship(scan, &executor);
    assert_eq!(
        shipped
            .schema()
            .field_with_name(row_number.name())
            .expect("the row-number column survives"),
        row_number.as_ref()
    );
    let batches = collect(shipped, executor.task_ctx()).await.unwrap();
    let mut positions: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(row_number.name())
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    positions.sort_unstable();
    // The test's INSERT writes one data file of six rows.
    assert_eq!(positions, (0..6).collect::<Vec<i64>>());
}

fn write_position_delete_file(
    path: &str,
    data_file_path: &str,
    partition: Struct,
    positions: &[i64],
) -> DataFile {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("extra", DataType::Int32, false),
        Field::new("renamed_position", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            POS_FIELD_ID.to_string(),
        )])),
        Field::new("renamed_path", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            FILE_PATH_FIELD_ID.to_string(),
        )])),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![42; positions.len()])),
            Arc::new(Int64Array::from(positions.to_vec())),
            Arc::new(StringArray::from(vec![data_file_path; positions.len()])),
        ],
    )
    .unwrap();

    let file = File::create(path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    let metadata = writer.close().unwrap();
    let file_size = std::fs::metadata(path).unwrap().len();

    DataFile::builder()
        .with_content(Content::PositionDeletes)
        .with_file_path(path.to_string())
        .with_file_format(FileFormat::Parquet)
        .with_partition(partition)
        .with_record_count(metadata.file_metadata().num_rows())
        .with_file_size_in_bytes(i64::try_from(file_size).unwrap())
        .with_column_sizes(None)
        .with_value_counts(None)
        .with_null_value_counts(None)
        .with_nan_value_counts(None)
        .with_distinct_counts(None)
        .with_lower_bounds(None)
        .with_upper_bounds(None)
        .build()
        .unwrap()
}

/// Creates `warehouse.test.orders` at `table_dir` in a fresh SQLite catalog,
/// registered with a new session. Unpartitioned when `partition_spec` is `None`.
async fn create_orders_table(
    table_dir: &str,
    schema: Schema,
    partition_spec: Option<PartitionSpec>,
    properties: &[(&str, &str)],
) -> (Arc<dyn Catalog>, SessionContext) {
    let object_store = ObjectStoreBuilder::Filesystem(Arc::new(LocalFileSystem::new()));
    let catalog: Arc<dyn Catalog> = Arc::new(
        SqlCatalog::new("sqlite://", "warehouse", object_store)
            .await
            .unwrap(),
    );
    catalog
        .create_namespace(&Namespace::try_new(&["test".to_string()]).unwrap(), None)
        .await
        .unwrap();
    let mut builder = Table::builder();
    builder
        .with_name("orders")
        .with_location(table_dir)
        .with_schema(schema);
    if let Some(partition_spec) = partition_spec {
        builder.with_partition_spec(partition_spec);
    }
    for (key, value) in properties {
        builder.with_property((key.to_string(), value.to_string()));
    }
    builder
        .build(&["test".to_owned()], catalog.clone())
        .await
        .unwrap();

    let ctx = SessionContext::new();
    ctx.register_catalog(
        "warehouse",
        Arc::new(IcebergCatalog::new(catalog.clone(), None).await.unwrap()),
    );
    (catalog, ctx)
}

/// Loads `warehouse.test.orders` and returns it with the path and partition
/// of its (single) live data file.
async fn load_orders_data_file(catalog: Arc<dyn Catalog>) -> (Table, String, Struct) {
    let identifier = Identifier::new(&["test".to_string()], "orders");
    let Tabular::Table(table) = catalog.load_tabular(&identifier).await.unwrap() else {
        panic!("orders should be an Iceberg table");
    };
    let manifests = table.manifests(None, None).await.unwrap();
    let data_files = table
        .datafiles(&manifests, None, (None, None))
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let (_, data_manifest_entry) = data_files
        .iter()
        .find(|(_, entry)| {
            entry.status() != &Status::Deleted && entry.data_file().content() == &Content::Data
        })
        .unwrap();
    let data_file_path = data_manifest_entry.data_file().file_path().clone();
    let partition = data_manifest_entry.data_file().partition().clone();
    (table, data_file_path, partition)
}

#[tokio::test]
async fn applies_v2_position_deletes() {
    let temp_dir = TempDir::new().unwrap();
    let table_dir = format!("{}/test/orders", temp_dir.path().display());

    let schema = Schema::builder()
        .with_struct_field(StructField {
            id: 1,
            name: "id".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::Long),
            doc: None,
            initial_default: None,
            write_default: None,
        })
        .with_struct_field(StructField {
            id: 2,
            name: "payload".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::String),
            doc: None,
            initial_default: None,
            write_default: None,
        })
        .with_struct_field(StructField {
            id: 3,
            name: "__data_file_path".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::Long),
            doc: None,
            initial_default: None,
            write_default: None,
        })
        .with_struct_field(StructField {
            id: 4,
            name: "__iceberg_file_row_position".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::String),
            doc: None,
            initial_default: None,
            write_default: None,
        })
        .with_struct_field(StructField {
            id: 5,
            name: "__iceberg_data_sequence_number".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::String),
            doc: None,
            initial_default: None,
            write_default: None,
        })
        .with_struct_field(StructField {
            id: 6,
            name: "category".to_string(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::String),
            doc: None,
            initial_default: None,
            write_default: None,
        })
        .build()
        .unwrap();

    let partition_spec = PartitionSpec::builder()
        .with_partition_field(PartitionField::new(
            6,
            1000,
            "category",
            Transform::Identity,
        ))
        .build()
        .unwrap();

    let (catalog, ctx) = create_orders_table(&table_dir, schema, Some(partition_spec), &[]).await;

    run_query(
        "INSERT INTO warehouse.test.orders VALUES
            (1, 'one', 101, 'row-one', 'seq-one', 'a'),
            (2, 'two', 102, 'row-two', 'seq-two', 'a'),
            (3, 'three', 103, 'row-three', 'seq-three', 'a'),
            (4, 'four', 104, 'row-four', 'seq-four', 'a'),
            (5, 'five', 105, 'row-five', 'seq-five', 'a'),
            (6, 'six', 106, 'row-six', 'seq-six', 'a')",
        &ctx,
    )
    .await;

    let (mut table, data_file_path, partition) = load_orders_data_file(catalog).await;

    let delete_dir = format!("{table_dir}/data");
    std::fs::create_dir_all(&delete_dir).unwrap();
    let delete_files = vec![
        write_position_delete_file(
            &format!("{delete_dir}/position-delete-1.parquet"),
            &data_file_path,
            partition.clone(),
            &[1, 4],
        ),
        write_position_delete_file(
            &format!("{delete_dir}/position-delete-2.parquet"),
            &data_file_path,
            partition,
            &[4, 5],
        ),
    ];

    table
        .new_transaction(None)
        .append_delete(delete_files)
        .commit()
        .await
        .unwrap();

    #[cfg(feature = "proto")]
    row_number_column_survives_shipping(
        "SELECT id, payload FROM warehouse.test.orders",
        &ctx,
        &table_dir,
    )
    .await;

    let batches = run_query(
        "SELECT id, payload FROM warehouse.test.orders ORDER BY id",
        &ctx,
    )
    .await;
    assert_batches_eq!(
        [
            "+----+---------+",
            "| id | payload |",
            "+----+---------+",
            "| 1  | one     |",
            "| 3  | three   |",
            "| 4  | four    |",
            "+----+---------+",
        ],
        &batches
    );

    // Shipped to an executor, the plan applies the same deletes. (The table's
    // paths have no scheme, so they are unchanged by normalization; pruning's
    // normalization is covered by `codec::tests`.)
    #[cfg(feature = "proto")]
    assert_batches_eq!(
        [
            "+----+---------+",
            "| id | payload |",
            "+----+---------+",
            "| 1  | one     |",
            "| 3  | three   |",
            "| 4  | four    |",
            "+----+---------+",
        ],
        &shipping::execute_shipped(
            &ctx,
            "SELECT id, payload FROM warehouse.test.orders ORDER BY id",
            &table_dir,
        )
        .await
    );

    // Queries that project no columns count the surviving rows.
    let batches = run_query("SELECT count(*) AS n FROM warehouse.test.orders", &ctx).await;
    assert_batches_eq!(["+---+", "| n |", "+---+", "| 3 |", "+---+",], &batches);
    let batches = run_query("SELECT 1 AS one FROM warehouse.test.orders LIMIT 1", &ctx).await;
    assert_batches_eq!(
        ["+-----+", "| one |", "+-----+", "| 1   |", "+-----+",],
        &batches
    );

    let batches = run_query(
        "SELECT __iceberg_data_sequence_number, id, __data_file_path,
                __iceberg_file_row_position
         FROM warehouse.test.orders ORDER BY id",
        &ctx,
    )
    .await;
    assert_batches_eq!(
        [
            "+--------------------------------+----+------------------+-----------------------------+",
            "| __iceberg_data_sequence_number | id | __data_file_path | __iceberg_file_row_position |",
            "+--------------------------------+----+------------------+-----------------------------+",
            "| seq-one                        | 1  | 101              | row-one                     |",
            "| seq-three                      | 3  | 103              | row-three                   |",
            "| seq-four                       | 4  | 104              | row-four                    |",
            "+--------------------------------+----+------------------+-----------------------------+",
        ],
        &batches
    );

    let batches = run_query(
        "SELECT id FROM warehouse.test.orders WHERE id IN (2, 3, 5) ORDER BY id",
        &ctx,
    )
    .await;
    assert_batches_eq!(
        ["+----+", "| id |", "+----+", "| 3  |", "+----+",],
        &batches
    );

    run_query(
        "INSERT INTO warehouse.test.orders VALUES
            (7, 'seven', 107, 'row-seven', 'seq-seven', 'a'),
            (8, 'eight', 108, 'row-eight', 'seq-eight', 'a')",
        &ctx,
    )
    .await;
    let batches = run_query(
        "SELECT id FROM warehouse.test.orders WHERE id >= 5 ORDER BY id",
        &ctx,
    )
    .await;
    assert_batches_eq!(
        ["+----+", "| id |", "+----+", "| 7  |", "| 8  |", "+----+",],
        &batches
    );
}
