import collections
from typing import List, Tuple, Dict

import json
import regex as re
from collections.abc import Iterator,Iterable


def get_stats(vocab_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
        辅助函数：统计相邻对的频率
        输入：整数列表 [1, 2, 1, 2, 3]
        输出：字典 {(1, 2): 2, (2, 3): 1}
        """
    counts = collections.defaultdict(int)
    #统计相邻元素对的频率
    for ids,freq in vocab_counts.items():
        #遍历相邻对
        for i in  range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] += freq
    return counts

def merge_ids(vocab_counts: Dict[Tuple[int, ...], int], pair:Tuple[int,int], idx) ->Dict[Tuple[int, ...], int]:
    """
    辅助函数：将列表中的 pair 替换为 idx
    输入：ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=256
    输出：[256, 3, 256]
    """
    new_vocab_counts = {}
    p0,p1 = pair
    for ids,freq in vocab_counts.items():
        #如果词中不包含 p0，则一定不需要合并
        # 正确写法：先保留原样，再 continue
        if p0 not in ids:
            new_vocab_counts[ids] = freq  # <--- 这行代码绝对不能少！
            continue
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
                #将合并操作后的新id列表转换为元组,在新建的词汇表字典中，记录这个合并后的序列及其频率
        # ✅ accumulate freq (GPT-2 correct)
        key = tuple(new_ids)
        new_vocab_counts[key] = new_vocab_counts.get(key, 0) + freq
    return new_vocab_counts

#区别merge，用于BPETokenizer
def _merge_ids_in_list(ids: list[int], pair: Tuple[int, int], new_id: int) -> list[int]:
    """在 id 列表中把每一处 pair 替换为 new_id，用于 bpe_ 推理。"""
    p0, p1 = pair
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes],  ##分词器词汇表，一个从 int（词汇表中标记的 ID）到 bytes（标记字节）的映射
                 merges: list[tuple[bytes, bytes]],  ##BPE 合并列表。列表中的每个元素都是一个字节元组 (<token1>, <token2>)，
                 special_tokens: list[str] | None = None  ##分词器使用的特殊字符串标记列表。这些字符串永远不会被拆分成多个标记，始终保持为一个标记。
                 ):
        # 建立正向词汇表 (self.vocab)：将传入的 vocab 复制一份保存，用于 decode。
        self.vocab = vocab
        # 建立反向词汇表 (self.id_to_token)：创建一个从 ID 到 token 的映射，用于 encode。
        self.encode_vocab = {v: k for k, v in vocab.items()}
        # 处理特殊标记 (special_tokens)：
        # 如果不为空，遍历 special_tokens。检查每个特殊标记是否已经在 vocab 中。如果不在，计算一个新的 ID（当前最大 ID + 1），将其编码为 bytes 后加入 self.vocab。
        self.special_tokens = special_tokens if special_tokens else []
        sorted_vocab = sorted(vocab.keys())  # 获取 vocab 中的 ID 并排序
        # 找到当前最大的id号，后续进行追加
        max_id = sorted_vocab[-1] if sorted_vocab else 0
        for token in self.special_tokens:
            # 将特殊字符转为bytes
            st_bytes = token.encode('utf-8')
            if st_bytes not in self.encode_vocab:
                max_id += 1
                self.vocab[max_id] = st_bytes
                self.encode_vocab[st_bytes] = max_id
        # 建立 BPE 合并规则表 (self.bpe_ranks)：
        # 我们需要将其转换为一个字典：{(bytes1, bytes2): rank}。
        self.bpe_ranks = {}
        for rank, (p1_bytes, p2_bytes) in enumerate(merges):
            # 还需要找到这两个合并后变成了什么 ID
            merges_bytes = p1_bytes + p2_bytes
            if merges_bytes in self.encode_vocab:
                new_id = self.encode_vocab[merges_bytes]
                # 存入字典：键是 (id1, id2)，值是 (rank, new_id)
                # 这样我们在 bpe_ 阶段既能比较优先级，又能直接拿到替换的 ID
                self.bpe_ranks[(p1_bytes, p2_bytes)] = (rank, new_id)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """从磁盘读取 vocab 与 merges 文件，组装成 Tokenizer 实例。"""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            file_json = json.load(f)
        vocab = {}
        for key, value in file_json.items():
            # GPT-2 等格式多为 {"token_str": id}，即 value 为 id
            idx = int(value)
            token_bytes = key.encode("utf-8") if isinstance(key, str) else key
            vocab[idx] = token_bytes
        #第二步：解析 Merges 文件 (Text -> List)
        #打开文件，按行读取
        merge = []
        with open(merges_filepath,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()#移除字符串开头和结尾的空白字符
                #跳过注释或者空行
                if not line or line.startswith('#'):
                    continue
                #分割字符串
                parts = line.split()
                if len(parts) == 2:
                    token1 = parts[0].encode('utf-8')
                    token2 = parts[1].encode('utf-8')
                    merge.append((token1, token2))
        return cls(vocab, merge, special_tokens)

    def bpe_(self, ids: list[int]) -> list[int]:
        """
        对给定的 id 列表执行 BPE 合并。
        bpe_ranks 的 key 是 (bytes, bytes)，需用 self.vocab 把 id 转为 bytes 再查表。
        """
        while len(ids) > 1:
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            # 用 (id_i, id_j) -> (rank, new_id)，查表时 id 转 bytes：self.vocab[id]
            #pairs是[int,int]，而self.bpe_ranks是(bytes, bytes) ，要查找是否存在，要先经过vocab转化
            exiting = {}
            for (id_a, id_b) in pairs:
                ba, bb = self.vocab.get(id_a), self.vocab.get(id_b)
                if (ba, bb) in self.bpe_ranks:
                    exiting[(id_a, id_b)] = self.bpe_ranks[(ba, bb)]
            if not exiting:
                break
            # 选 rank 最小的 pair；(pair_ids, (rank, new_id))
            (pair_ids, (rank, new_id)) = min(exiting.items(), key=lambda x: x[1][0])
            ids = _merge_ids_in_list(ids, pair_ids, new_id)
        return ids

    # 将单个字符串编码为 ID 列表。
    def encode(self, text: str) -> list[int]:
        # 对特殊字符进行转义（如 | -> \|），并用 | 连接成正则
        special = self.special_tokens
        if special:
            pattern = "(" "|".join(re.escape(tok) for tok in special) + ")"
            text_segments = re.split(pattern, text)
        else:
            text_segments = [text]
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 存放编码后的字节
        encode_sequence = []
        for segment in text_segments:
            if not segment:
                continue
            if segment in special:
                encode_sequence.append(self.encode_vocab[segment.encode('utf-8')])
                continue
            # 对当前 segment 做预分词再 BPE
            words = re.findall(PAT, segment)
            for word in words:
                # 将单词转为字节
                word_bytes = word.encode('utf-8')
                # 将字节拆分为单个字节的 ID 列表
                word_ids = list(word_bytes)
                # 词被拆分为字节序列
                """例
                word = " world"
                word_encoded = b" world"
                bytes_list = [32, 119, 111, 114, 108, 100]
                """
                # 对该字节 ID 列表执行 BPE 合并
                bpe_ids = self.bpe_(word_ids)
                # 将结果添加到最终的编码序列中
                encode_sequence.extend(bpe_ids)
        return encode_sequence

    def encode_iterable(self, iterable: Iterable[str]  # 比如一个打开的文件句柄，或者一个字符串列表
                        ) -> Iterator[int]:
        # 遍历输入的可迭代对象
        for text_chunk in iterable:
            yield from (self.encode(text_chunk))

    def decode(self, ids: list[int]) -> str:
        token_bytes = bytearray()
        # 拼接对应的bytes
        for ids_id in ids:
            if ids_id in self.vocab:
                token_bytes.extend(self.vocab[ids_id])
        return token_bytes.decode("utf-8", errors="replace")




