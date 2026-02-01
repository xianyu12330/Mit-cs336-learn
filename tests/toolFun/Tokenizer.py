import collections
from typing import List, Tuple, Dict
import regex as re
from collections import Counter


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
        new_vocab_counts[tuple(new_ids)] = freq
    return new_vocab_counts


class BPETokenizer:
    #初始化
    def __init__(self, vocab:dict[int, bytes],#分词器词汇表，一个从 int（词汇表中标记的 ID）到 bytes（标记字节）的映射
                 merges:list[tuple[bytes, bytes]],#BPE 合并列表。列表中的每个元素都是一个字节元组 (<token1>, <token2>)，
                 special_tokens:list[str] | None = None #分词器使用的特殊字符串标记列表。这些字符串永远不会被拆分成多个标记，始终保持为一个标记。
                 ):
        self.vocab = vocab.copy()
        #反向编码表：bytes标记字节-》词汇表id的映射
        self.vocab_encoder = {} #dict[bytes,int]
        for key,value in vocab.items():
            self.vocab_encoder[value] = key
        #2.处理特殊字符
        #需要更新 vocab 和 vocab_encoder，为特殊标记分配新 ID
        self.special_tokens = special_tokens if special_tokens  else []
        sorted_vocab = sorted(vocab.keys())
        #找到当前最大的id号，后续进行追加
        max_id = sorted_vocab[-1] if sorted_vocab else 0

        for st in special_tokens:
            # 将特殊标记转为 bytes 处理 (假设输入是 utf-8)
            st_bytes = st.encode("utf-8")
            #判断特殊字符的比特是否存在，不存在则新添加
            if st_bytes not in self.vocab_encoder:
                self.vocab_encoder[max_id] = st_bytes
                # 同时更新正向 vocab
                vocab[max_id] = st_bytes
                max_id += 1
        #构建bpe合成规则
        # 我们需要将 merges 里的 (bytes, bytes) 转换为 (int, int) -> (rank, new_id)
        self.bpe_ranks = {}
        for rank,(p1_bytes,p2_bytes) in enumerate(merges):
            # 核心修正：将字节对转为 ID 对
            if p2_bytes in self.vocab_encoder and p1_bytes in self.vocab_encoder:
                p1_id =  self.vocab_encoder[p2_bytes]
                p2_id = self.vocab_encoder[p1_bytes]
                merge_bytes = p1_bytes + p2_bytes
                # 还需要找到这两个合并后变成了什么 ID
                if merge_bytes  in self.vocab_encoder:
                    new_id = self.vocab_encoder[merge_bytes]
                # 存入字典：键是 (id1, id2)，值是 (rank, new_id)
                # 这样我们在 bpe_ 阶段既能比较优先级，又能直接拿到替换的 ID
                    self.bpe_ranks[(p1_id,p2_id)] = (rank,new_id)

    #实现迭代合并bpe,函数负责处理一段已经转换为字节 ID 列表的普通文本。
    def bpe_(self,ids: list[int])-> list[int]:
        """
        对给定的 id 列表执行 BPE 合并。
        这里可以使用之前写好的 merge_ids 函数。
        """
        #保证有至少两个字节的id能进行合并，若只有一个id则无法合并
        while len(ids) >= 2:
            #获取所有相邻对
            pairs = [(ids[i],ids[i + 1]) for i in range(len(ids) - 1)]
            #要找的是：在 self.bpe_ranks 中存在，且 rank 值最小的那个 pair
            exiting_pair = {pair :self.bpe_ranks[pair] for pair in pairs if pair in self.bpe_ranks}
            #如果在字典内没有该配对，说明合并该结束
            if not exiting_pair:break
            # 找到 rank 最小（优先级最高）的 pair
            # min 的 key 参数用于根据字典的值（rank）来比较
            min_pair = min(exiting_pair, key=lambda k: exiting_pair[k][0])
            #将合并后的配对插入字典
            new_id = self.bpe_ranks[min_pair]
            #替换可配对的元素
            ids = merge_ids(ids,min_pair,new_id)
        return ids

    #将文本编码为整数id
    def tokenize_encode(self,text:str,special_tokens:list[str] | None = None) -> List[int]:
        # 1. 确保 special_tokens 是一个集合，方便快速查找
        if special_tokens is None:
            special_tokens = []
        special_tokens_set = set(special_tokens)
        if special_tokens :
            # 对特殊字符进行转义（如 | -> \|），并用 | 连接成正则
            # 结果类似: "\<\|endoftext\|\>|\<\|pad\|\>"
            pattern ="(" "|".join(re.escape(tok) for tok in special_tokens) + ")"
            text_segments = re.split(pattern, text)
        else:
            text_segments = [text]
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        #存放编码后的字节
        encode_segments = []
        for text in text_segments:
            #跳过空字符
            if not text:continue
            #如果是特殊字符，将特殊字符对应的词表id插入，因为后续id转为字符仍然需要特殊字符
            if text in special_tokens:
                special_encode = text.encode("utf-8")
                if special_encode in self.vocab_encoder:
                    encode_segments.append(self.vocab_encoder[special_encode])
                else:
                    print(f"Warning: Special token {text} not in vocab")
                continue
            #对片段进行编码
            words = re.findall(PAT,text)
            for word in words:
                word_encoded = word.encode("utf-8")
                # 第二步：将字节转换为整数列表 (初始 ID 序列)
                bytes_list = list(word_encoded)
                #词被拆分为字节序列
                """例
                word = " world"
                word_encoded = b" world"
                bytes_list = [32, 119, 111, 114, 108, 100]
                """
                merge_list = self.bpe_(bytes_list)
                encode_segments.append(merge_list)
        return encode_segments

    #将整数id解码回文本
    def tokenize_decode(self,voc_list:list[int])->str:
        token_bytes = bytearray()
        #拼接对应的bytes
        for voc_id in voc_list:
            if voc_id in self.vocab:
                token_bytes.extend(self.vocab[voc_id])
            else:
                print(f"Warning: ID {voc_id} not found in vocab")
        return token_bytes.decode("utf-8",errors="replace")


    #返回tokenizer文本编码器
    def get_tokenizer(vocab,merges,special_tokens = None):
        return BPETokenizer(vocab,merges,special_tokens)

