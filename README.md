[toc]
# NLP-course
## 项目说明

This project is using for master course <Natural Language Processing> of NEU, China.
本项目是东北大学计算机科学与工程学院研究生一年级课程《自然语言处理》的结课作业。

作者：刘沛灼 2101762

## 文件说明

本项目共计4个文件，即4个小的demo，存放在demo文件夹中，所有文件可以在Google Colab \ Kaggle上运行：

*   GRU-Seq2Seq.ipynb

    本demo通过GRU实现了一个英文单词“反义词”的序列转换，如：

    ```python
    man up    -> women down 
    black man  -> white women
    black king  -> white duwn 
    high fat   ->  low thin  
    girl man   ->  boy women 
    high man   ->  low women 
    man small  ->  women big 
    left king  -> right women
    ```
    其中的数据是自己所写，数据量较小，验证了一下模型的有效性。本demo也是最主要完成的demo，相关报告内容见ipynb文件内部描述。模型的checkpoint保存为model.ckpt。
*   HW4.ipynb
    本demo是台湾大学李宏毅教授2022年春《机器学习》课程的课程作业4，相关要求和slides在slides文件夹中，本项目通过使用了transformer的encoder，完成了一个语音辨识的分类任务，相关数据集在kaggle上，可以在kaggle的比赛[here](https://www.kaggle.com/competitions/ml2022spring-hw4)中运行该demo。demo由于是课程作业，有一定的原始代码存在，需要完成的部分是TODO部分。
*   HW5.ipynb
    本demo是台湾大学李宏毅教授2022年春《机器学习》课程的课程作业5，相关要求和slides在slides文件夹中，本项目通过使用transformer，完成了英文到繁体中文的翻译任务，相关数据集在使用的是TED2020演讲的翻译文本，由社区志愿者翻译，可以在[这里](https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz)和[这里](https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz)进行下载。demo由于是课程作业，有一定的原始代码存在，需要完成的部分是TODO部分。
*   transformer-implement(half-finished).ipynb
    本demo是一个transformer的复现，用以对WMT2014的英-德语料进行翻译，参考了[这里的实现](http://www.egbenz.com/#/my_article/49)但由于时间关系，并没有完全理解着写完，还存在一些错误，还没能完全跑通，目前手动改进的地方是对原始文章中的greedysearch改进为了beamsearch：
    ```python
    import heapq
    class Beam:
        def __init__(self, beam_width):
            self.heap = list()  # 存储各个beam search的结果
            self.beam_width = beam_width  # beam的数量

        def add(self, probility, complete, seq, decoder_input, decoder_hidden):
            """
            添加数据，同时判断总的数据个数，多则删除
            :param probility: 概率乘积
            :param complete: 最后一个是否为EOS
            :param seq: list，所有token的列表
            :param decoder_input: 下一次进行解码的输入，通过前一次获得
            :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
            :return:
            """
            heapq.heappush(self.heap, [probility, complete,
                                    seq, decoder_input, decoder_hidden])
            # 判断数据的个数，如果大，则弹出。保证数据总个数小于等于beam_width
            if len(self.heap) > self.beam_width:
                heapq.heappop(self.heap)

        def __iter__(self):  # 让该beam能够被迭代
            return iter(self.heap)


    def beam_decode(model, src, src_mask, max_len, start_symbol, BEAM_SIZE):
        model.eval()
        beam_seq = Beam(BEAM_SIZE)
        # 构造第一次需要的输入数据，保存在堆中
        memory = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        beam_seq.add(1, False, [ys], ys, out)
        while True:
            cur_beam = Beam(BEAM_SIZE)
            # 取出堆中的数据，进行forward_step的操作，获得当前时间步的output，hidden
            for _probility, _complete, _seq, _decoder_input, _decoder_hidden in beam_seq:
                # 判断前一次的_complete是否为True，如果是，则不需要forward
                if _complete == True:
                    cur_beam.add(_probility, _complete, _seq,
                                _decoder_input, _decoder_hidden)
                else:
                    decoder_hidden = model.decode(
                        memory, src_mask, _decoder_input, subsequent_mask(ys.size(1)).type_as(src.data))
                    decoder_output_t = model.generator(decoder_hidden[:, -1])
                    value, index = torch.topk(decoder_output_t, BEAM_SIZE)
                    # 从output中选择topk（k=beam width）个输出，作为下一次的input
                    for m, n in zip(value, index):
                        decoder_input = torch.LongTensor([[n[0]]])
                        decoder_input = torch.cat(
                            [_decoder_input, decoder_input], dim=1)
                        seq = _seq + [n[0]]
                        probility = _probility * m[0]
                        # probility = _probility + m
                        if n[0].item() == 1:  # index of </s>
                            complete = True
                        else:
                            complete = False
                        cur_beam.add(probility, complete, seq,
                                    decoder_input, decoder_hidden)
            # 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否是EOS结尾或者是否达到最大长度，如果是，停止迭代
            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            if best_complete == True or len(best_seq) - 1 == max_len:  # 减去</s>
                seq = [i.item() for i in best_seq]
                return seq
                # return best_seq
            else:
                # 重新遍历新的堆中的数据
                beam_seq = cur_beam

    model_out = beam_decode(model, rb.src, rb.src_mask, 72, 0, BEAM_SIZE=3)
    ```
    本项目中的beamsearch方法也以小组合作的形式，用进了本学期另一门课程《深度学习及其应用》中。



