### 代码结构

```
code/
├── config.py                   # 配置管理
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖列表
├── rag_modules/               # 核心模块
│   ├── __init__.py
│   ├── data_preparation.py    # 数据准备模块
│   ├── index_construction.py  # 索引构建模块
│   ├── retrieval_optimization.py # 检索优化模块
│   └── generation_integration.py # 生成集成模块
└── vector_index/              # 向量索引缓存（自动生成）

```

### 数据准备模块 data_preparation.py

**（系统初始化时运行）**

- 利用pathlib的Path类，递归的遍历数据目录下所有的md文件。为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径），封装为langchain_core.documents import Document对象
- 对每个doc，进行元数据增强（感觉就是增加元数据metadata包含的数据量，比如菜品名称、分类、难度）。主要是进行菜品名称、分类、难度的提取
  - 分类推断: 从HowToCook项目的目录结构推断菜品分类
  - 难度提取: 从内容中的星级标记自动提取难度等级
  - 名称提取: 直接使用文件名作为菜品名称
- 将完整的菜谱文档按照Markdown标题结构进行分块，实现父子文本块架构。 
  其中包含：

```
("#", "主标题"),      # 菜品名称
("##", "二级标题"),   # 必备原料、计算、操作等
("###", "三级标题")   # 简易版本、复杂版本等
```

- 主要是使用Markdown分割器，主要就是按标题结构进行分割，分隔个数为主标题、二级标题、三级标题的个数之和。每个子块都会继承父文档的元数据并加上自己的元数据，并建立父子映射关系
- **为什么要切块？**
  - embedding 模型有最大 token 限制
  - 小块语义更集中
  - 相似度检索更准确

**（用户询问时运行）**

- 当用户询问"宫保鸡丁怎么做"时，可能会检索到同一道菜的多个子块。把检索到的子块对应的父块去重，并按检索到子块的数量进行排序，返回一个父块的列表
- 去重逻辑：
  - 统计相关性: 计算每个父文档被匹配的子块数量
  - 按相关性排序: 匹配子块越多的菜谱排名越靠前
  - 去重输出: 每个菜谱只输出一次完整文档

### 索引构建 index_construction.py

使用Faiss，Faiss的全称是[Facebook AI Similarity Search](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/faiss)，是FaceBook针对大规模相似度检索问题开发的一个工具。本地高性能向量数据库。**FAISS 内部逻辑是：暴力扫描，对内部所有向量进行相似度计算，精度 100%，适合 N ≤ 几百万。**

索引构建流程：

```
原始文档
  ↓（文本切分）
chunks（List[Document]）
  ↓（Embedding 模型）
向量（List[float]）
  ↓（FAISS 索引）
vectorstore（相似度搜索）
```
做了三件事：

1. **遍历每个 chunk 的 `page_content`**
2. **调用 embedding 模型，把文本转成向量**  每个chunk就是一个向量
3. **把所有向量和对应的 chunk 存进 FAISS 索引**

LangChain 在 FAISS 外面“包”了什么？

它是一个 三件套结构：

1️⃣ FAISS Index（只管向量）

```
self.index  # faiss.IndexFlatL2
```

2️⃣ Docstore（存 Document）

默认是：

```
InMemoryDocstore
```


结构非常朴素：

```
self._dict = {
    "uuid-1": Document(...),
    "uuid-2": Document(...),
    ...
}
```

3️⃣ index_to_docstore_id（关键映射）

这是你问的 “id → Document 映射表” 的真实形态：

```
index_to_docstore_id = {
    0: "uuid-1",
    1: "uuid-2",
    2: "uuid-3",
    ...
}
```

查询时实际流程：

```
query
 ↓
embed_query(query)
 ↓
query_vector
 ↓
FAISS 相似度搜索（L2 / Cosine）
 ↓
返回最相近的 k 个向量
 ↓
映射回对应的 chunk Document
```

### 检索优化 retrieval_optimization.py

RRF算法综合两种检索方式：BM25检索和向量检索。比如最后要选出4个最相关的chunk，先由BM25选出10个，再由向量检索选出10个，然后合并得到20个候选集合。然后使用rrf算法，综合这两个分数对chunk进行重排序，选出最高的的4个。

```text
Query
 ├── BM25（关键词召回）
 └── Vector（语义召回）
      ↓
   合并 / 重排（Rerank）
      ↓
   Top-k chunks → LLM
```

BM25 的打分来自三种直觉：

1️⃣ 词出现得越多 → 越相关（但有上限）

- 一个词在文档中出现 1 次很重要
- 出现 10 次 ≠ 重要性 ×10（会“饱和”）

2️⃣ 越少见的词 → 权重越大（IDF）

- `"the"` 几乎没信息量
- `"beamforming"` 非常有区分度

👉 这就是 **IDF（逆文档频率）**

3️⃣ 文档太长要惩罚（长度归一化）

- 长文档更容易“误中”关键词
- BM25 会对**过长文档降权**

**在当前系统中，两种检索方式各有优势：**

**向量检索的优势**：

- 理解语义相似性，如"简单易做的菜"能匹配到标记为"简单"的菜谱
- 处理同义词和近义词，如"制作方法"和"做法"、"烹饪步骤"
- 理解用户意图，如"适合新手"能找到难度较低的菜谱

**BM25检索的优势**：

- 精确匹配菜名，如"宫保鸡丁"能准确找到对应菜谱
- 匹配具体食材，如"土豆丝"、"西红柿"等关键词
- 处理专业术语，如"爆炒"、"红烧"等烹饪手法

**采用RRF进行混合检索**

**RRF 的公式**
$$

\mathrm{RRF}(d) = \sum_{i=1}^{n} \frac{1}{k + \mathrm{rank}_i(d)}

$$
参数解释

| 符号     | 含义                           |
| -------- | ------------------------------ |
| ddd      | 文档                           |
| nnn      | 检索器数量（BM25、Vector、…）  |
| ranki(d) | 文档 d 在第 i 个检索器中的排名 |
| kkk      | 平滑常数（常用 **60**）        |

RRF 不看分数，只看“排名”。

**元数据过滤检索**

按照**元数据中已有的信息**比配相似度，如菜品分类，难度等。

比如：

- 用户询问"推荐几道素菜"时，可以按菜品分类过滤，只检索素菜相关的内容
- 新手用户问"有什么简单的菜谱"时，可以按难度等级过滤，只返回标记为"简单"的菜谱
- 想做汤品时询问"今天喝什么汤"，可以按分类过滤出所有汤品菜谱

**最终送给 LLM 的 Prompt 结构**

```
System:
You are a domain expert...

Context:
[1] chunk A
[2] chunk B
[3] chunk C

Question:
<user query>

Answer using only the context.
```

### 集成模块 generation_integration.py

实现了一个查询路由，分为三类查询：

1.'list' - 用户想要获取菜品列表或推荐，只需要菜名   例如：推荐几个素菜、有什么川菜、给我3个简单的菜 

2.'detail' - 用户想要具体的制作方法或详细信息   例如：宫保鸡丁怎么做、制作步骤、需要什么食材 

3.'general' - 其他一般性问题   例如：什么是川菜、制作技巧、营养价值
