## NLP

#### chendanqi
(https://www.cs.princeton.edu/~danqic/papers/thesis.pdf)
```
part of speech tagging ：断句，知道名词，副词，动词
name entity recognition 实体识别，识别人名 地名
syntactic parsing: 句法分析，理解单词之间的关系，或者句子结构
coreference resolution: 代词借代 it 指谁， the girls 指谁

用阅读理解来测试机器是否读懂文章
reading comprehension
阅读理解的成果主要取决于以下两个原因
1.大量监督学习数据库（passage，question, answer）
2.神经网络阅读理解的发展（公式化问题，建造模型，系统主要成分，优劣）

神经网络阅读理解的两个方向
1.open-domain question answering:  检索和阅读理解的 联合挑战（例如回答问题百科wiki）
2.conversational question answering：对话和阅读理解的结合，多会和问题回答

chapter2
reading comprehension的综述
1.作为一个监督学习任务，4种类别
2.reading comprehension 和 question answering的区别：他们的goals不同

机器学习的方法（2013-2015）
 f : (passage, question) ?→ answer.
以前的那些模型的缺点
1.依赖已有的语言工具，例如parsers 和 semantic role labeling,这种现有工具都是一个领域模型训练出来的（比如说是新闻类）
杠杆化使用这种标注，会给ml模型带来噪声。
更高阶的标注会get worse。（eg discourse relastions vs part of speech tagging）

2.模仿人类级别的理解很难，很难从当下的语句表达中 构造有效的特征
3.人类标记的量太小了，不足以支撑庞大的统计模型


ml的复兴
SQUAD数据库

Differing from feature-based classifiers, neural reading
comprehension models have several great advantages
1.不依赖于下游特征，避免语言标注带来的噪音，自己学自己
2.传统的nlp特征模型，有个缺点：特征太少；数据太少，
也不好校正特征的权重
3.神经网络不需要构建特征模型，所以它本身概念上简单，也能更专注于设计神经架构。tensorflow 和 tytorch太nb了，有他们开发新模型很快~

nlp在squad数据库上面表现很好，但是不能说明解决了阅读理解的问题，因为squad的数据例子都很简单，不需要复杂的理解

后面就又更多的数据库，例子很复杂，多种阅读理解挑战

2.2 任务定义
1.公式问题
a passage of text p and a corresponding question q
answer a as output
f : (p, q) −→ a

四种阅读理解模式
填空 CNN/DAILY MAIL
选择 MCTEST
从文中选择答案 SQUAD
自由回答 d NARRATIVEQA 

填空和选择 评价方式看重精确
从文中选择答案 另一种评价方式
em 对了1错了0
f1 计算预测和正确答案的重叠率
F1 =(2 × Precision × Recall)/(Precision + Recall)
标点忽略不计，a an the 忽略不计

自由回答时
用标准衡量矩阵 来检测NLG过程（NLG比如说翻译，总结，BLEU，Meteor，ROUGE等）

2.3 阅读理解 vs 问题回答
final goals不一样
回答问题是基于知识库 回答人类提出的问题；（如果识别区分相关资料，如何整合知识，如何学习什么是人类常见问题）
阅读理解 是用问题衡量阅读理解，所以问题答案来自于text本身。

阅读理解对构建问题回答系统也有帮助

2.4 数据与模型

chapter3

3.1以前的模型 特征提取模型
特征模型是
the correct answer 阿尔法
is expected to rank higher than all other candidate entities:

w转置fp,q(a) > w转置fp,q(e), ∀e ∈ E \ {a},

特征向量
 a feature vector fp,q(e) ∈ R

p = passage 
q= question

每个实体e都构建了特征向量后，我们就可以用ml算法（svm 逻辑回归 ）

特征选取的时候，很定制化  （频率啊，位置啊 是不是问句 ; 和段落问题的关系啊 距离啊 ; 依赖现有工具 pos parse的标记精度等）
feature eg:
e是否在问题里
e是否在段落里
e频率
e的第一个位置
word distance
sentence co-occurrence
n-gram exact match
dependency parse match
别人的研究成果里研究了各种feature

3.2 一个神经网络的方法 the astanford attentive reader
基础知识
1.word embedding
高维向量表示
Vcar = [0, 0, 0, 0 , ,,,,,, 1,,,,,,,0]转置
两个词之间向量无关 cos（Va,Vb）=0
低维会减轻这个问题 cos(car vechicle)<cos(car, man)

假设 words出现在相似的上下文 趋近 有相似的意义 - 》word embeeding 可以从无标记语料库中 有效学习。
一些对外发布的版本，例如：word2vec,glove,fasttext.
这些是现代nlp系统的主流。

2.循环神经网络 Recurrent neural networks

RNN使用可变长度的序列
ht = f(ht−1, xt; Θ)

对于nlp应用，我们把一句话或者一段路作为一个word序列，每个单词都是一个向量（通常被word embedding训练过）
Vanilla RNNs take the form of
ht = tanh(Whh*ht−1 + Whx*xt + b)

ht ∈ Rh 是模型
Whh ∈ Rh×h
,Whx ∈ Rh×d
, b ∈ Rh
是要学习的参数

为了优化RNN模型，各种方法都被提出来了
LSTM，GRU 比较有名。
LSTM最有竞争力。

rnn有更细致一点的，双向rnn

3.attention mechanism 注意机制
第一次出现于 seq2seq模型，后来用于nlp

主要是，我们想预测句子的情感，或者翻译句子，我们通常用rnn给单句编码：h1,h2,h3,...hn，并且用hn来预测情感或者目标语言的第一个单词。
这需要模型能把一个句子中所有必要信息压缩到一个固定长度的向量里。这是个瓶颈。
注意力机制，比起把所有信息压缩到last hidden vector，它随时着眼于hidden vector，选择一个合适的subset of vectors。

大概来说，注意力机制就是给每个hi算个分，再分类器始终返回一个离散分布

近期，有人说注意力机制不需要和rnn一起使用，rnn可以纯粹基于word embedding 和前向反馈网络，与此同时提供最小的序列信息。
这种模型参数少，可大规模并行运行，
transformer模型 也是一个趋势。

3.2.2 模型

模型综述
首先build a vector 代表问题，passage的每个token一个vector
计算问题和passage word的相似函数
用question-passage 相似分数来决定开始和结束位置。
低维度，pre-train的word embedding 给每个段落和问题的单词。
编码和相似函数共同优化最后的答案预测。

细节
question encoding
问题的每个word 先embedding，再bi-lstm
这些hidden unit 再通过attention机制变成一个向量
（我们发现 加这一层注意力layer 有用，因为它给相关问题words加了更多权重）

passage encoding
先对序列进行bi-lstm
序列有两种，一种是passage的每个单词，一种是和问题的相关性。
第一种的话，先把单词embedding了，我们也加了一些手工特征（pos，NER named entity recognition tags TF term frequency ）
pos 和 ner 用现有工具转化，因为tag是很小。
tf是测量words在段落里出现多少次

第二种的话，有两种表现方式
1.em exact match。作者用了三种binary features：original，lowercase，lemma form
2.aligned question embedding: 相近单词之间的联系 比如car 和 vehicle

p˜i = (femb(pi), ftoken(pi), fexact match(pi), falign(pi)) ∈ Rd˜

回答预测 answer prediction
注意力机制
两个分类器，一个预测回答start，一个预测回答end

训练和推理
训练目标是最小化交互熵loss
参数由随机梯度优化

3.2.3 拓展
讲了几个模型，都是得到向量o，用o来完成预测（预测 完形填空，单选等）
o的形成形式各有不同，但输入都是pi q。


```

ACL2019最佳论文，共八篇文章获奖。其中包含一篇最佳长论文、一篇最佳短论文、五篇杰出论文、一篇最佳 Demo 论文。

#### 弥补神经机器翻译在训练和推理过程之间的缺口
(https://arxiv.org/pdf/1906.02448.pdf)
 * 该论文解决了seq2seq转换中长期存在的暴露偏差问题；
 * 论文所提出的解决方案是：判断依据在“基于参考文本中的词”和“解码器自己的输出中预选择词”两种之间切换
 * 这个方法适用于当前的teacher-forcing训练范式，并改进了规划抽样；
 * 该方法也适用于其他seq2seq任务。
 * 论文的实验做的非常完善，结果令人信服，并可能影响机器翻译未来的工作；
 * 论文摘要：神经机器翻译（NMT）是以上下文为条件来预测下一个词，从而顺序地生成目标词。在训练时，它以ground truth词汇作为上下文进行预测；而在推理时，它必须从头开始生成整个序列。反馈上下文信息的这种差异会导致误差累积。此外，词级训练要求所生成的序列与ground truth序列之间严格匹配，这导致对不同的但合理的翻译的过度校正。在本文中，我们在模型训练中不仅从ground truth序列还从预测序列中来采样上下文，其中预测序列是用句子级最优来选择的。我们在Chinese->English 和 WMT'14 English->German的翻译任务的实验结果表明，我们的方法可以在多个数据集上实现显著的改进。

#### “你知不知道佛罗伦萨全都是游客？”，评价最先进的说话人承诺模型
(https://www.aclweb.org/anthology/P19-1412)
  * 对基于规则的和双向LSTM这两种最先进的说话人承诺模型进行了系统的评价
  * 有语言学知识的模型
  * 论文摘要：当一个人，比如 Mary，问你「你知不知道佛罗伦萨全都是游客？」，我们会认为她相信佛罗伦萨全都是游客；但如果她问「你觉得佛罗伦萨游客多吗？」，我们就不会这样认为。推断说话人承诺（或者说事件真实度）是问答和信息提取任务中的关键部分。在这篇论文中，作者们探索了这样一个假说：语言学信息的缺乏会影响说话人承诺模型中的错误模式。他们的验证方式是在一个有挑战性的自然语言数据集上分析模型错误的语言学关联性。作者们在 CommitmentBank 这个由自然英语对话组成的数据集上评价了两个目前最好的说话人承诺模型。CommitmentBank 数据集已经经过了说话人承诺标注，方式是在 4 种取消蕴含的环境中向着时态嵌入动词（比如知道、认为）的补充内容进行标注。作者们发现，一个带有语言学知识的模型能展现比基于 LSTM 的模型更好的表现，这表明如果想要在这样的有挑战性的自然语言数据中捕捉这些信息的话，语言学知识是必不可少的。对语言学特征的逐项分析展现出了不对称的错误模式：虽然模型能在某些状况下得到好的表现（比如否定式），但它很难泛化到更丰富的自然语言的语言学结构中（比如条件句式），这表明还有很大提升的空间。
  
#### 情绪-原因对的提取：文本情感分析中的一个新任务
(https://arxiv.org/pdf/1906.01267.pdf)
 * 在文本中通过联合学习来识别情感及原因
 * 提出一个新的有趣的模型：两种不同类型的多任务架构，一种是任务独立的，另一种是交互的。
 * 根据相互作用的方向，实现情绪（精确度）或原因（召回）的改善。
 * 论文摘要：情绪原因提取（Emotion cause extraction ，ECE）是一项旨在提取文本中某些情绪背后潜在原因的任务，近年来由于其广泛的应用而受到了很多关注。然而，它有两个缺点：1）情绪必须在ECE原因提取之前进行标注，这极大地限制了它在现实场景中的应用；2）先标注情绪然后提取原因的方式忽略了它们是相互指示的事实。在这项工作中，我们提出了一项新任务：情绪 - 原因对提取（emotion-cause pair extraction ，ECPE）。这个任务旨在提取文本中潜在的情绪-原因对。我们提出了两步法来解决这个新的ECPE任务。首先通过多任务学习单独地进行的情绪提取和原因提取，然后进行情绪-原因配对和过滤。基准情绪-原因语料库的实验结果证明了ECPE任务的可行性以及我们方法的有效性。


#### 文本摘要重要性的一个简单的理论模型
(https://www.aclweb.org/anthology/P19-1101)
 * 这篇文章讨论了自动文本摘要中长期存在的深层问题：如何衡量摘要内容的适用性？
 * 提出了「内容重要性」的三部分理论模型
 * 提出了建设性的评估指标
 * 文章中还与标准指标和人类判断进行了比较
 * 论文摘要：摘要研究主要由经验方法驱动，手工精心调制的系统在在标准数据集上表现良好，但其中的信息重要性却处于隐含状态。我们认为建立重要性（Importance）的理论模型会促进我们对任务的理解，并有助于进一步改进摘要系统。为此，我们提出了几个简单但严格定义的概念：冗余（Redundancy），相关性（Relevance）和信息性（Informativeness）。这些概念之前只是直观地用于摘要，而重要性是这些概念统一的定量描述。此外，我们提供了建议变量的直观解释，并用实验证明了框架的潜力以知道后续工作。

#### 用于面向任务的对话系统的可传输的多领域状态生成器
(https://arxiv.org/pdf/1905.08743.pdf)
 * 本文解决了传统但未解决的问题：对话状态跟踪中看不见的状态；表明可以从用户话语中生成对话状态；
 * 新方法可扩展到大值集（large value sets）并能处理以前看不见的值；
 * 除了展示最先进的结果外，本文还研究了针对新领域的few-shot学习。
 * 论文摘要：过度依赖领域本体和缺乏跨领域知识共享是对话状态跟踪的两个实际存在但研究较少的问题。现有方法通常在在推理过程中无法跟踪未知slot 值，且通常很难适应新领域。在本文中，我们提出了一个可转换对话状态生成器（Transferable Dialogue State Generator，TRADE）它使用复制机制从话语中生成对话状态，当预测在训练期间没有遇到的（domain，slot，value）三元组时可以促使知识转移。我们的模型由一个话语编码器、slot gate、状态生成器组成，它们跨域共享。实验结果表明，TRADE在人类对话数据集MultiWOZ的五个领域中实现了最先进的联合目标准确率48.62%。此外，我们通过模拟针对未见过的领域的zero-shot和few-shot对话状态跟踪，证明了其传输性能。在其中一个zero-shot域中TRADE实现了60.58%的联合目标准确率，并且能够适应少数几个案例而不会忘记已经训练过的域。


#### 我们需要谈谈标准的数据集分割做法
(https://wellformedness.com/papers/gorman-bedrick-2019.pdf)
 * 本文质疑了评估NLP模型时公认且广泛运用的方法；
 * 本文提出了几种关于数据集的标准拆分方法；
 * 本文使用POS标记说明了问题；
 * 本文建议系统排名应当基于使用随机分组的重复评估方法
 * 论文摘要：语音和语言技术的标准做法是根据在一个测试集上的性能来对系统进行排名。然而很少有研究人员用统计的方法来测试性能之间的差异是否是由偶然原因造成的，且很少有人检查同一个数据集中分割出不同的训练-测试集时的系统排名的稳定性。我们使用了2000年至2018年间发布的九个词性标注器进行复现实验，这些标注器每个都声称在广泛使用的标准的分割方式上获得了最佳性能。然而当我们使用随机生成的训练-测试集分割时，根本无法可靠地重现某些排名。我们在此建议使用随机生成的分割来进行系统比较。
 
 #### 通过阅读实体描述进行零样本实体链接
 (https://arxiv.org/pdf/1906.07348.pdf)
  * 本文提出了一种新颖的词义消歧系统，专门用于提高稀少的和未见过的词上的表现；
  * 本文提出的感知选择任务被视为连续任务，并且使用了资源的组合；
  * 本文的结果富有洞察力，并且改善了现有水平。
  * 论文摘要：我们提出了zero-shot实体链接任务，其中mentions必须链接到没有域内标记数据的未曾见过的实体。这样做的目的是实现向高度专业化的领域的鲁棒迁移，也因此我们不会假设有元数据或别名表。在这种设置中，实体仅通过文本描述进行标记，并且模型必须严格依赖语言理解来解析新实体。首先，我们表明对大型未标记数据进行预训练的阅读理解模型可用于推广到未曾见过的实体。其次，我们提出了一种简单有效的自适应预训练策略，我们将其称为域自适应预训练（domain-adaptive pre-training ，DAP），DAP可以解决与在新域中链接未见实体的域迁移问题。我们在为此任务构建的新数据集上进行的实验，显示了DAP在强预训练基线（包括BERT）上有所改进。本文提供了数据集和代码。


#### OpenKiwi: An Open Source Framework for Quality Estimation (一个qe开源框架)
(https://arxiv.org/pdf/1902.08646.pdf)
 * 这是机器翻译中第一个可以自由使用的用于执行基于神经的质量估计的框架；
 * 包含了WMT 2015-18基准评估中四种最佳质量评估系统的实现；
 * 包含了易于使用的API和可复现的实验
 * 论文摘要：我们介绍基于PyTorch的开源框架OpenKiwi，这个框架可用于翻译质量评估。OpenKiwi支持单词级和句子级质量评估系统的训练和测试，实现了WMT 2015-18 质量评估比赛中的最佳系统。我们在WMT 2018（英-德 SMT 和NMT）的两个数据集上对OpenKiwi进行了基准测试，在单词级任务达到最先进的水平，句子级任务中也能够接近最先进的水平。
