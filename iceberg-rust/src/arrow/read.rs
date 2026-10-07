/*!
 * Functions to read arrow record batches from an iceberg table
*/

use std::{convert, ops::Range, sync::Arc};

use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::{future::BoxFuture, stream, FutureExt, Stream, StreamExt, TryFutureExt};
use object_store::{path::Path, ObjectStore, ObjectStoreExt};
use parquet::{
    arrow::{
        arrow_reader::ArrowReaderOptions, async_reader::AsyncFileReader,
        ParquetRecordBatchStreamBuilder,
    },
    errors::{ParquetError, Result as ParquetResult},
    file::metadata::{ParquetMetaData, ParquetMetaDataReader},
};

use crate::error::Error;
use crate::object_store::store::path_from_location;

use iceberg_rust_spec::spec::manifest::{FileFormat, ManifestEntry};

// A minimal [`AsyncFileReader`] over a data file in an [`ObjectStore`].
//
// Iceberg always knows a data file's size up front (it is tracked in the
// manifest entry), so unlike a general-purpose object-store reader this
// implementation never needs to fall back to suffix range requests to
// locate the Parquet footer.
// TODO: Consider wrapping this with `parquet::arrow::async_reader::SpawnedReader` at some point.
pub(crate) struct DataFileReader {
    object_store: Arc<dyn ObjectStore>,
    path: Path,
    file_size: u64,
}

impl DataFileReader {
    pub(crate) fn new(object_store: Arc<dyn ObjectStore>, path: Path, file_size: u64) -> Self {
        Self {
            object_store,
            path,
            file_size,
        }
    }
}

impl AsyncFileReader for DataFileReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        self.object_store
            .get_range(&self.path, range)
            .map_err(|err| ParquetError::External(Box::new(err)))
            .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        async move {
            self.object_store
                .get_ranges(&self.path, &ranges)
                .await
                .map_err(|err| ParquetError::External(Box::new(err)))
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        async move {
            let file_size = self.file_size;
            let metadata = ParquetMetaDataReader::new()
                .with_metadata_options(options.map(|o| o.metadata_options().clone()))
                .load_and_finish(self, file_size)
                .await?;
            Ok(Arc::new(metadata))
        }
        .boxed()
    }
}

/// Read a parquet file into a stream of arrow recordbatches. The record batches are read asynchronously and are unordered
pub async fn read(
    manifest_files: impl Iterator<Item = ManifestEntry>,
    object_store: Arc<dyn ObjectStore>,
) -> impl Stream<Item = Result<RecordBatch, ParquetError>> {
    stream::iter(manifest_files)
        .then(move |manifest| {
            let object_store = object_store.clone();
            async move {
                let data_file = manifest.data_file();
                match data_file.file_format() {
                    FileFormat::Parquet => {
                        let object_reader = DataFileReader::new(
                            object_store,
                            path_from_location(data_file.file_path())
                                .map_err(|err| Error::External(Box::new(err)))?,
                            (*data_file.file_size_in_bytes()) as u64,
                        );
                        Ok::<_, Error>(
                            ParquetRecordBatchStreamBuilder::new(object_reader)
                                .await?
                                .build()?,
                        )
                    }
                    format => Err(Error::NotSupported(format!(
                        "reading data file {} of format {format:?}",
                        data_file.file_path()
                    ))),
                }
            }
        })
        .map(|result| match result {
            Ok(batches) => batches.left_stream(),
            Err(err) => stream::iter(std::iter::once(Err(ParquetError::External(Box::new(err)))))
                .right_stream(),
        })
        .flat_map_unordered(None, convert::identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg_rust_spec::spec::{
        manifest::{Content, DataFile, Status},
        table_metadata::FormatVersion,
        values::Struct,
    };
    use object_store::memory::InMemory;

    fn data_entry(path: &str, format: FileFormat) -> ManifestEntry {
        let data_file = DataFile::builder()
            .with_content(Content::Data)
            .with_file_path(path.to_string())
            .with_file_format(format)
            .with_partition(Struct {
                fields: vec![],
                lookup: Default::default(),
            })
            .with_record_count(0)
            .with_file_size_in_bytes(0)
            .with_column_sizes(None)
            .with_value_counts(None)
            .with_null_value_counts(None)
            .with_nan_value_counts(None)
            .with_distinct_counts(None)
            .with_lower_bounds(None)
            .with_upper_bounds(None)
            .build()
            .unwrap();
        ManifestEntry::builder()
            .with_format_version(FormatVersion::V2)
            .with_status(Status::Added)
            .with_data_file(data_file)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn invalid_manifest_location_is_reported_instead_of_skipped() {
        let entry = data_entry("s3://bucket/data//broken.parquet", FileFormat::Parquet);
        let stream = read(std::iter::once(entry), Arc::new(InMemory::new())).await;
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 1, "a bad file must not silently disappear");
        assert!(items[0].is_err());
    }

    #[tokio::test]
    async fn unsupported_format_error_names_the_file_and_format() {
        let entry = data_entry("s3://bucket/data/deletes.puffin", FileFormat::Puffin);
        let stream = read(std::iter::once(entry), Arc::new(InMemory::new())).await;
        let items: Vec<_> = stream.collect().await;
        let error = items[0].as_ref().unwrap_err().to_string();
        assert!(
            error.contains("s3://bucket/data/deletes.puffin") && error.contains("Puffin"),
            "{error}"
        );
    }
}
