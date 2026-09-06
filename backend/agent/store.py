"""
Two small in-memory stores. Neither is a cross-part contract — they exist
because Part 2 owns the API surface and the job queue needs somewhere to
resolve `image_ids` (from POST /api/query) back to the ImageMetadata that
POST /api/upload produced earlier.

ImageStore: image_id -> ImageMetadata, populated by every successful upload.

UploadSessionStore: tracks in-progress chunked uploads (Section 3.2 hardening:
"Resumable chunked handling on /api/upload for large files"). This is
deliberately simple — Section 7 explicitly permits "tus or basic chunking",
not a from-scratch protocol: the client sends `total_chunks` + `chunk_index`
+ an `upload_id` (self-picked on the first chunk), the server writes each
chunk to its own file and reports back which indices it has, and a client
that dropped mid-upload can resume by re-sending only the chunks missing
from that list — no separate status endpoint needed. In-memory is enough
for a single-process demo deployment (Section 7: "single-deployment demo");
a real multi-worker deployment would back this with shared storage instead.

Swapped for a real datastore during integration if needed — nothing outside
this file knows these are in-memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from shared.schemas import ImageMetadata, Modality


class ImageStore:
    def __init__(self) -> None:
        self._images: dict[str, ImageMetadata] = {}

    def put(self, metadata: ImageMetadata) -> None:
        self._images[metadata.image_id] = metadata

    def get(self, image_id: str) -> Optional[ImageMetadata]:
        return self._images.get(image_id)


class UploadSession:
    def __init__(self, upload_id: str, total_chunks: int, modality: Modality,
                 timestamp: Optional[str], dir_path: Path) -> None:
        self.upload_id = upload_id
        self.total_chunks = total_chunks
        self.modality = modality
        self.timestamp = timestamp
        self.dir_path = dir_path
        self.received: set[int] = set()
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def write_chunk(self, index: int, data: bytes) -> None:
        (self.dir_path / f"chunk_{index:06d}").write_bytes(data)
        self.received.add(index)

    def is_complete(self) -> bool:
        return len(self.received) == self.total_chunks

    def assemble(self) -> Path:
        final_path = self.dir_path.with_suffix(".assembled")
        with open(final_path, "wb") as out:
            for i in range(self.total_chunks):
                out.write((self.dir_path / f"chunk_{i:06d}").read_bytes())
        return final_path


class UploadSessionStore:
    def __init__(self, base_dir: Path) -> None:
        self._sessions: dict[str, UploadSession] = {}
        self._base_dir = base_dir

    def get_or_create(self, upload_id: str, total_chunks: int, modality: Modality,
                       timestamp: Optional[str]) -> UploadSession:
        if upload_id not in self._sessions:
            self._sessions[upload_id] = UploadSession(
                upload_id, total_chunks, modality, timestamp, self._base_dir / upload_id
            )
        return self._sessions[upload_id]

    def discard(self, upload_id: str) -> None:
        self._sessions.pop(upload_id, None)
