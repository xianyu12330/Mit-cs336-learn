import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,#二进制文件
    desired_num_chunks: int,#希望切成 多少份
    split_special_token: bytes,#用来作为 合法切分点的特殊 token
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # 确保你在byte层操作，避免 .find() 时隐式编码错误
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes计算文件大小（byte 级）
    file.seek(0, os.SEEK_END)
    file_size = file.tell()#返回 字节数
    file.seek(0)

    #理论上的初始 chunk 大小
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    #初始化 chunk 边界
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    #分块向前扫描
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    #修正边界，第一个边界 0：固定，最后一个边界 file_size：固定，中间的才需要修正

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    # 不同chunk可能：找到同一个 <|endoftext|>
    return sorted(set(chunk_boundaries))


## Usage
with open(..., "rb") as f:
    num_processes = 4
    # 寻找边界
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    # 串行处理
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
