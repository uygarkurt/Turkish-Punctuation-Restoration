# Transformer Based Punctuation Restoration Models for Turkish

With the latest development in mobile devices and social networks, many users share their opinions, daily routines or news. Speed of the communication is getting more important but the quality of the text is getting less. One of the most common mistake in the texts is not putting punctuations to correct places in it. This process can be called punctuation restoration in this paper. The well-known application of the punctuation restoration is automatic speech recognition. In automatic speech recognition, the text is generated from the voice and the task has two important challenges to write sentences. The first challenge is to generate grammatically correct sentences. The second challenge is to restore punctuations correctly.    

Automatic Speech Recognition (ASR) has become a widely used application. One of the major drawbacks of the ASR is that it leaves the transformed text unpunctuated. Manually punctuating large amounts of text is time and resource consuming. Apart from automatically transformed texts, human generated texts may end up with missing or wrongly used punctuations. Even though this topic is studied in English language extensively it's not the case for the Turkish language. This paper introduces a punctuation restoration models by fine-tuning pre-trained transformers and a Turkish punctuation restoration dataset for Turkish.

Scores of the models can be summarized as below:

|   Model  |  PERIOD  |          |          |   COMMA  |          |          | QUESTION |          |          |  OVERALL |          |          |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          |     P    |     R    |    F1    |     P    |     R    |    F1    |     P    |     R    |    F1    |     P    |     R    |    F1    |
|   BERT   | 0.972602 | 0.947504 | 0.959952 | 0.576145 |  0.70001 | 0.632066 | 0.927642 | 0.911342 |  0.91942 | 0.825506 | 0.852952 | 0.837146 |
|  ELECTRA | 0.972602 | 0.948689 | 0.960497 |  0.5768  | 0.710208 |  0.63659 | 0.920325 | 0.921074 | 0.920699 | 0.823242 | 0.859990 | 0.839262 |
| ConvBERT | 0.972731 | 0.946791 | 0.959585 | 0.576964 | 0.708124 | 0.635851 | 0.922764 | 0.913849 | 0.918285 | 0.824153 | 0.856254 | 0.837907 |


Dataset can be summarized as below:

|    Split    |  Total  | Period (.) | Comma (,) | Question (?) |
|:-----------:|:-------:|:----------:|:---------:|:------------:|
|    Train    | 1471806 |   124817   |   98194   |     9816     |
| Validation  |  180326 |    15306   |   11980   |     1199     |
|   Test      |  182487 |    15524   |   12242   |     1255     |
