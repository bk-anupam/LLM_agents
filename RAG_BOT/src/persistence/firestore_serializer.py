import base64
import gzip
import gzip
from langgraph.checkpoint.serde.base import SerializerProtocol

class FirestoreSerializer:
    def __init__(self, serde: SerializerProtocol):
        self.serde = serde

    def dumps_typed(self, obj):
        type_, data = self.serde.dumps_typed(obj)
        # Compress data before encoding
        compressed_data = gzip.compress(data)
        data_base64 = base64.b64encode(compressed_data).decode('utf-8')
        return type_, data_base64

    def loads_typed(self, data):
        type_name, serialized_obj = data
        decoded_data = base64.b64decode(serialized_obj.encode('utf-8'))
        try:
            # Try to decompress, assuming new compressed format
            decompressed_data = gzip.decompress(decoded_data)
        except (gzip.BadGzipFile, OSError):
            # If decompression fails, assume it's old, uncompressed data
            decompressed_data = decoded_data
        return self.serde.loads_typed((type_name, decompressed_data))

    def dumps(self, obj):
        data = self.serde.dumps(obj)
        # Compress data before encoding
        compressed_data = gzip.compress(data)
        data_base64 = base64.b64encode(compressed_data).decode('utf-8')
        return data_base64

    def loads(self, serialized_obj):
        decoded_data = base64.b64decode(serialized_obj.encode('utf-8'))
        try:
            # Try to decompress, assuming new compressed format
            decompressed_data = gzip.decompress(decoded_data)
        except (gzip.BadGzipFile, OSError):
            # If decompression fails, assume it's old, uncompressed data
            decompressed_data = decoded_data
        return self.serde.loads(decompressed_data)
