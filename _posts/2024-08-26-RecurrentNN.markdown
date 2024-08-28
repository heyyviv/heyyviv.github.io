---
layout: post
title: "Recurrent Neural Networks"
date: 2024-08-26 01:43:18 +0530
categories: Deep Learning
---

To go from multilayer networks to recurrent networks, we need to take advantageof one of the early ideas found in machine learning and statistical models of the1980s: sharing parameters across diﬀerent parts of a model. Parameter sharingmakes it possible to extend and apply the model to examples of diﬀerent forms(diﬀerent lengths, here) and generalize across them. If we had separate parametersfor each value of the time index, we could not generalize to sequence lengths notseen during training, nor share statistical strength across diﬀerent sequence lengthsand across diﬀerent positions in time. Such sharing is particularly important whena speciﬁc piece of information can occur at multiple positions within the sequence.

“I went to Nepal in 2009” and “In 2009,I went to Nepal.” If we ask a machine learning model to read each sentence andextract the year in which the narrator went to Nepal, we would like it to recognizethe year 2009 as the relevant piece of information, whether it appears in the sixth word or in the second word of the sentence.
A traditional fully connectedfeedforward network would have separate parameters for each input feature, soit would need to learn all the rules of the language separately at each position inthe sentence. By comparison, a recurrent neural network shares the same weightsacross several time steps.

![RNN](/assets/rnn_1.jpg)
![RNN](/assets/rnn_2.jpg)

When the recurrent network is trained to perform a task that requires predictingthe future from the past, the network typically learns to use h(t) as a kind of lossysummary of the task-relevant aspects of the past sequence of inputs up tot. Thissummary is in general necessarily lossy, since it maps an arbitrary length sequence(x(t), x(t−1), x(t−2), . . . , x(2), x(1)) to a ﬁxed length vectorh(t). Depending on the training criterion, this summary might selectively keep some aspects of the pastsequence with more precision than other aspects.
![RNN](/assets/rnn_4.jpg)
![RNN](/assets/rnn_5.jpg)
![RNN](/assets/rnn_6.jpg)
![Seq2Seq](/assets/rnn_3.jpg)
