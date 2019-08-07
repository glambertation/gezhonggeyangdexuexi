## NLP

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