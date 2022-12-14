# Unsupervised Emotion Clustering using Multimodal Features on a Personal Language Corpus

### Introduction

Each individual produces a vast amount of language daily, data that shelters insights into their defense mechanisms, patterns of emotional response, and lives as a whole. However, these corpuses of language remain largely unexplored. Though recent years have seen a spike in research on human affect recognition and emotion detection (ED), little research has been done to personalize this technology for individuals looking to use these ED tools as therapeutic methods of self-reflection. 

Most approaches to ED in the literature consist of supervised classification models which rely on prescriptive emotional labels (happy, sad, angry, and neutral). Moreover, these labeled data are often produced by a homogenous group of actors in less a than organic setting. This approach makes several problematic assumptions about emotional affect in language. What does it mean to classify emotions into artificially prescribed categories when emotions are continuous, dimensional, and ambiguous? Moreover, can the idiosyncratic experiences of an individual's affect be classified using such non-personalized and inorganic data?

In the light of the great differences in emotional response that individuals emote and the time-consuming and prescriptive process of labeling language data, it is necessary to develop new unsupervised methods for ED that are non-prescriptive. In this paper, I present a new unsupervised approach using the K-Means algorithm and demonstrate its efficacy on my own personal language corpus. Given that we can extract lexical and acoustic features from a vocal signal as measurable correlates to emotion, any unsupervised method we propose relies on the answer to the critical question: 

**Does proximity in feature space translate to proximity in emotion space?**

### Related Work

Speech supplies its listener with a rich representation of the communicator’s emotional state in both its acoustic and lexical features. But while humans can identify the latent emotional state of an utterance with relative ease, the nuances of a voice’s emotional content is rather difficult for a machine to capture. I will begin by reviewing various self-monitoring systems in the psychological domain, then discuss the use of acoustic features to classify mental illnesses, and finally examine multimodal approaches to ED. 

The mental health crisis attending the pandemic and the prior decade has spawned a remarkable surplus of technology geared towards the recognition, prevention, and alleviation of mental illness. Demand has soared with over 70% of patients showing interest in using mobile apps to self-monitor their mental health (Torous et al., 2014). Many have turned towards popular interactive applications like Headspace and Woebot to get mental health support, approaches which have shown demonstrable positive outcomes for many users (Mani et al., 2015; Fitzpatrick et al., 2017). Other self-monitoring approaches are journal-like in their approach, requiring patients to log their mood and complete surveys daily to connect them with swift, responsive treatment. This approach was taken by Bardram et al. (2013) in the MONARCA self-assessment system to track early warning signs of bipolar shifts between manic and depressive episodes and by Rohani et al. (2019) in the delivery of behavioral activation therapy to depressed individuals via routine self-assessment on an app. 

Self-assessment tools rely heavily on a patients self-recollected and self-perceived behavior for diagnosis and treatment. Grünerbl et al. (2014) and many others argue that this approach lacks objectivity, while also creating a self-monitoring paradigm that is too heavily predicated on the discipline and willingness of the patient to fill out daily surveys ad infimum. In response, the authors propose a passive monitoring system using acoustic features, location, and acceleration collected via smart-phone sensors to detect manic-depressive swings in bipolar patients, inferring their mental states in an autonomous manner. Despite the authors’ innovative approach to patient monitoring using both acoustic features and portable technology (two key ideas in my system), they fail to make use of the semantically-rich lexical features attending the patients’ vocal signal. 

This focus on acoustic features over lexical content is not isolated to this paper but is pervasive in much psychological research. Clinicians have used vocal features in mental health examinations since Newman and Mather (1938) recognized distinct patterns of vocal response attending various affective disorders, like the “pressured” speech of bipolar patients or “monotone" and “lifeless” speech of depressive patients. More recently, these intuitions have been the basis of various classification algorithms that seek to identify mental illness via the "biological" markers in the voice. These algorithms use low-level descriptors (LLD) of the vocal signal to classify everything from PTSD to Alzheimer’s with some success (Marmar et al., 2019; Fraser et al., 2016). Unfortunately, in pursuit of an "objective" biological signal or metric with which to diagnose mental illness—as if it were as easy as an X-ray detection of a broken bone—computational psychological research has largely ignored the predictive potential of the voice’s concurrent lexical information. 

ED algorithms proposed by natural language and signal processing researchers often incorporate both acoustic and lexical features into their multimodal models. Lexical feature representations have varied across the years, from the use of key-words and phrase rules (Cowie et al., 1999), semantic trees Zhe and Boucouvalas (2002), NGrams (Polzin and Waibel, 2000), and vector space models Schuller et al. (2005). The multimodal model proposed by Jin et al. (2015) which was later expanded by Gamage et al. (2017) was particularly successful. In the paper, the authors select the top 500-600 words from each of the four emotion classes represented in the IEMOCAP database using the classic tf-idf weighting method (Busso et al., 2008; Salton and McGill, 1986). In addition to these BOW lexical features, the authors propose a new "eVector" 4-dimensional feature to better capture the emotional content of an utterance that weigh words that occur exclusively in certain emotional contexts higher than those that appear uniformly across contexts so as to infuse the model with the emotional salience of words in the lexicon. The attempt to compute emotional salience of particular words hearkens back to the rule-based approach of Cowie et al. (1999), but their method is limited to the existence of pre-labeled emotional classes. To uncover the emotional salience of particular words in unsupervised clustering, another approach is required. 

Deep-neural networks (DNN) have since replaced many previous feature generation schemes in ED tasks. One example is the multimodal model proposed by Kim and Shin (2019) trained on the same IEMOCAP data. In addition to replacing the BOW features with the semantically rich 300dimensional word2vec features from the seminal Mikolov et al. (2013) paper, the authors capture the emotional content of words using a large affective lexicon which maps words onto a 3-D VAD space (valence, arousal, dominance) to model greater intricacies of emotional salience (A. B. Warriner and Brysbaert, 2013). One potential area of improvement was suggested by Guo et al. (2022) who applied transformers to better model the interdependence of lexical and acoustic information in an utterance.

Despite the innovative methods of feature encoding that these classification models put forth, there are several issues and assumptions endemic to the IEMOCAP data which brings their results into question: (1) the emotional categories of "happy", "sad", "angry", and "neutral" don’t reflect the diversity of emotions that we experience, (2) the situations the actors performed were fabricated from the emotion, rather than emerging from phenomenological experience, creating stereotypical conceptions of said emotion, and (3) the conception of what it means to be "happy", "sad", "angry", or "neutral" are specific to the language/identity of the actors, their communities in Southern California, and the particular days in 2008 in which they were spoken (Busso et al., 2008). In the attempt to generalize their results and manufacture stable targets for their algorithms, this data does not demonstrate the power of ED algorithms in the real world. For the use case of emotional self-monitoring and self-understanding, the data is insufficient in size, insufficient in emotional intricacy, and insufficient in its generality to deliver any real benefits to an individual looking to monitor their mind. Part of the contribution of my paper is a personalized approach to data collection that contextualizes emotions in the life of the individual. By hyper-localizing the data to represent the idiosyncratic emotions of a single person, this paper then explores clustering as a possibility to develop semantically meaningful emotional classes in a non-prescriptive manner, combating the reductionist approach that classification algorithms take to this continuous space. 


### Methods

#### Data

A core feature of my proposed system is the use of a personal corpus to uncover hyper-local representations of emotional response. This exploration is fueled by data that I have collected over the past 10 months through a lavalier microphone. I envision a Bluetooth device in the future connected to a smartphone to improve ease of access and allow for a cloud-based interface, similar to the system proposed by Grünerbl et al. (2014). In total, the corpus is comprised of 160 hours of audio collected over 10 months, with its share of heartbreaks and irritations, outbursts and celebrations. To process this audio into text, I built a simple voice activity detector to isolate voiced segments in the audio and a random forest classification model to extract the time-segments in the audio corresponding to my speech as differentiated from others. These segments are then transcribed using Google's speech-to-text API and matched with its time segment in the original audio for acoustic feature extraction. Due to computational limitations, I isolated the months of April and May 2022 in the data for this study.

#### Feature Extraction

For the acoustic features, I use the PyAudioAnalysis library to extract the energy, energy entropy, spectral centroid, spectral rolloff, and zero-crossing rate for each 50 millisecond window of the audio. These features are then averaged together across each sentence's time segment to generate sentence-level acoustic feature representations. The acoustic features were chosen based on common sense, their higher relative variances in the data, and their good performance across ED literature (Weninger et al., 2013). For the lexical features, I will use the emotional salience vectors of Kim and Shin (2019) using the 10,000 common words which were annotated with VAD (valence, arousal, dominance) coordinates by A. B. Warriner and Brysbaert (2013). I average this "e-vector" across the utterance and weigh each word's contribution to the "document" of each audio transcript using the tf-idf weighting scheme of Salton and McGill (1986). This weighting process is essential so as to keep high frequency words like “like” out of the picture, and to make sure that low frequency words that are emotionally meaningful (like “fuck”) aren’t drowned out. The acoustic and the lexical feature vectors will then need to be concatenated. The feature vector will not be complete here, however, as each of these features have vastly different measures attached to them, so a normalization step is necessary before proceeding to clustering.


#### K-Means Clustering

In response to the limitations of current self-monitoring paradigms and the limited scope of emotion recognition algorithms, I use a clustering system to group similar emotional responses together. These clusters can be used to train classifiers or be analyzed by themselves. I will cluster the multimodal feature vectors using the k-means algorithm, which minimizes within-cluster variances (dubbed the "inertia") using squared Euclidean distances. As the ground truth labels are not known, the optimal number of clusters must be inferred quantitatively using the inertia and the silhouette coefficient, which is a measure that takes into account the mean distance between a sample and all other points in the same class and the mean distance between a sample and all other points in the next nearest cluster.

#### To PCA... or not to PCA

The Euclidean distance measure underlying the K-Means algorithm assumes that every feature dimension is weighted equally. However, since different emotions are elicited in different ways, each of our features might be more or less important for various emotion clusters. Principal Component Analysis (PCA) seems to be an answer to this dilemma, but there are benefits and drawbacks to its use.

Benefits of PCA:
- In finding the dimensions of maximum variance, PCA will reduce noise and guarantee similar dimensional weighting
- PCA makes the data easier to visualize
- PCA is unsupervised

Drawbacks of PCA:
- There is often very little information seperating crying from laughing, or peace from melancholy. In removing “insignificant” variance, PCA might flatten exactly the minute differences that are the desideratum for emotions of similar affect. 
- PCA reduces dimensional interpretability.

The solution I came up with is to make two clustering models. The first will not apply PCA prior to clustering, and will form clusters directly on the 8-dimensional feature vector. The second involves a reduction of the 8-dimensional feature vector down to two dimensions using PCA prior to clustering.

### Evaluation

#### Without PCA

![Figure 1: No-PCA Inertia Elbow](/elbow_inertia.png)

![Figure 2: No-PCA Silhouette Elbow](/elbow_silhouette.png)

#### Number of Clusters

Emotions are a continuous space in which there are no clear-cut "classes". Any cluster that is thereby formed within this emotional feature space serves only an organizational purpose. This ambiguity taken together with the high density of the data contributes to the difficulty of finding clear-cut classes using standard unsupervised evaluation tools. These lack of clear groupings are evident in Figure 2, which shows a silhouette coefficient that is positive but consistently low, indicating the presence of overlapping clusters. The inertia, a measure of internal coherency that represents the minimization of within cluster sums-of-squares, tells us that the the optimal number of clusters lie between 7 and 10, which is confirmed by the silhouette coefficient local maximum at 10. Though the silhouette coefficient is higher at 7 clusters, my aim is to capture more fine-grained emotional representations and thus 9 clusters serves this project better.

#### With PCA

![Figure 3: PCA Inertia Elbow](/elbow_inertia_PCA.png)

![Figure 4: PCA Silhouette Elbow](/elbow_silhouette_PCA.png)

As you can see, though the silhouette coefficient is significantly higher in the case of PCA, the bend in the elbow happens earlier (around 4-5 clusters) with the inertia and the silhouette graphs. In the lower dimensional space, there are fewer viable clusters in the feature space, as the data melds together. Additionally, in Figure 5 below, it is clear that the dimensions are far less interpretable. We will explore several methods of qualitative evaluation in the next section.

#### Qualitative Evaluation

It is difficult to evaluate unsupervised models as we are lacking "ground-truth" labels. This is the principle drawback to the approach, but also its principle strength. The point of this entire process is to build an emotion classification tool that is not dependent on developing a time-consuming and prescriptive emotional “labels” that do not generalize to individuals with different expressions of emotional affect. Any evaluation will necessarily be rather subjective, especially as we are dealing with a continuous space of emotions that cannot be clustered discretely with ease. Even if we did have clear-cut labels for the data—of intricate emotional complexity—the model would undoubtedly perform rather poorly on quantitative assessments due to the continuous nature of the emotion space. This can be seen by the meager silhouette scores of 0.30 and 0.12 (for PCA and no-PCA respectively) above. These two factors, when taken together, mean that there is no easy way to quantitatively evaluate whether the models have successfully classified utterances into one emotional class or another.

We must instead resort to a qualitative evaluation of each of the clusters to determine their quality. Let us begin with the 9-cluster No-PCA model. In Table 2, I chose 5 sentences at random from each cluster and located their corresponding audio. I then shuffled the sentences and cut off my own access to their corresponding cluster, and annotated each sentence with the perceived emotion that I drew from their audio stream blindly. I repeat this process three times in order to validate my assumptions about each class. 

Then, once the labeling had been completed, I whittled away confusing "half-thoughts" and utterances lacking any emotion (logistics) to get three examples per class. Once the logistical, unemotional language had been cleared away, the clustering performed relatively well. Cluster 0 corresponded to many of my long, inspired rants which all had high energy, high lexical affect. The sentences in this cluster tended to be longer, with an average sentence-length of 23 words (much higher than the corpus average of 11). Cluster 1 contained a mess of different emotions, and was among the poorest performing clusters. I believe this cluster is the product of high zero-crossing rates as all of the examples I drew had zero-crossing rates higher than 75\% of the rest of the data. This acoustic feature corresponds to high percussiveness and I will not include it on the next iteration of the K-Means. Most other clusters performed rather well. Cluster 3 was uniformly filled with curse words and slurred speech and represented some of my most distressed moments; cluster 4 was very uniformly disappointed and defeated; cluster 5 was vague, inflective, and uncertain; cluster 6 was more aggressive; cluster 7 was sad, cute, and pitiful; cluster 8 was annoyed; cluster 9 was generally positive and straight forward but as a whole was hard to differentiate from cluster 2. It is important to note that these evaluations are subject to confirmation bias and I hope to expand upon this evaluation in the future with a more thorough analysis (with other individuals labeling the perceived emotion rather than me).

Although I am using definitive emotions to describe each of these classes, things are a little more ambiguous in reality. If one were to focus purely on the emotionally indicative sentences in each cluster, many of them tended to have similar emotional qualities. But these islands of emotion lie in a sea of logistics and random chit-chat which had little to no emotional valence and were generally evenly split between the clusters. One possibility for future development is to first build a discriminator that decides whether an utterance is emotionally potent or not before clustering.

![Table 1: Certain clusters like cluster 1 had little to no coherence, with a mixture of different sentiments, emotions, and acoustic qualities, others performed very well like clusters 4, 7, and 8.](/Table2.png)

The PCA model was far more difficult to interpret along its axes as it had squashed the data and contained more general emotional classes. Please feel free to use my finished feature/sentence data (`cluster_sentence.csv`) and the `visualize_all.py` script to visualize the PCA graph interactively. This script allows you to mouse over the datapoints and recieve the corresponding sentences in the data. I haven't made my audio available due to size limitations and privacy concerns, but I am working on this...

![Figure 5: 5-cluster K-Means on PCA Features.](/PCA_KMeans.png)

To subjectively evaluate the graph, I looked at the sentence from top to bottom and from side to side, listening to the respective sentence-segments in the audio (a very time consuming process). The red cluster towards the bottom has extremely positive emotional affect associated with it, containing lots of "thank yous" and words like "gorgeous", "pretty" and "wonderful". The corresponding audio tended to reflect the valence of the words used, although it was far more vague. The north pole of the graph (the blue cluster) contains sentences like "I can go die in hell" and "Anything dark and gross scares me" or "Oh my back hurts, man". The acoustic affect associated with the top of the graph also reflected the valence of these sentences. This north-south axis (blue/red clusters respectively) seems to be semantically meaningful, representing positive (south) and negative (north) utterances, in both the lexical and the acoustic content of the sentences. This is a success, and indicates the importance of using multimodal feature-matrix for this classification task. Unfortunately, I was not able to understand/decode the east-west (purple, pink, green) utterances in the data. It seems to correspond with some of the acoustic features, as one moves from more noisy and incomprehensible utterances on the left of the graph (my mic picked up a lot of background noise) to more clear and quiet utterances on the right. This is not necessarily semantically meaningful, corresponding more with the quality of the audio stream rather than my emotional states (although one could argue that noisy contexts create more spirited and confused emotions, as opposed to the quiet contexts on the right). This leads me to believe that due to the noise of every-day audio streams, one either needs to implement a better microphone settup or use a voice-isolation ML tool to clean up the audio prior to processing, or one needs to weight the lexical features more heavily in the next iteration of the model.


### Limitations & Conclusion 

In this paper I perform unsupervised emotion clustering on a personal language corpus using the K-Means algorithm and demonstrate its efficacy using only VAD lexical features and a few emotionally rich acoustic features. However, this approach suffers from a few limitations. First, it is hard to evaluate models like these, and there is no performance guarantee on any individual's personal data set. Second, the language analyzed is limited to the 10,000 words that exist in the VAD corpus. This excludes any emotions that I express in my native language of Swedish. Third, due to the different seeding of the K-Means algorithm, the classes do not necessarily converge to the same groupings on each iteration. The lack of deterministic outcomes is also due to the continuous space of emotions, that do not fall into distinct classes. Even humans approaching the same data would have a hard time agreeing on the number or name of classes into which to classify these utterances. More testing must be done in the future using labeled corpuses in order to fully demonstrate this approaches' efficacy.

### Bibliography

V. Kuperman A. B. Warriner and M. Brysbaert. 2013. Norms of valence arousal and dominance for 13915 english lemmas. 45. 

Jakob Bardram, Mads Frost, Károly Szántó, Maria Faurholt-Jepsen, Maj Vinberg, and Lars Kessing. 2013. Designing mobile health technology for bipo- lar disorder: A field trial of the monarca system. Proceedings of the SIGCHI Conference on Human Factors in Computing Systems. 

Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jean- nette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan. 2008. Iemocap: interactive emotional dyadic motion capture database. 48. 

R. Cowie, E. Douglas-Cowie, B. Apolloni, J. Taylor, A. Romano, and W. Fellenz. 1999. What a neural net needs to know about emotion words. 

Kathleen Kara Fitzpatrick, Alison Darcy, and Molly Vierhile. 2017. Delivering cognitive behavior therapy to young adults with symptoms of depression and anxiety using a fully automated conversational agent (woebot): A randomized controlled trial. 4. 

Kathleen Fraser, Jed Meltzer, and Frank Rudzicz. 2016. Linguistic features identify alzheimer’s disease in narrative speech. 49. 

Kalani Wataraka Gamage, Vidhyasaharan Sethu, and Eliathamby Ambikairajah. 2017. Salience based lex- ical features for emotion recognition. 

Theodoros Giannakopoulos. 2015. pyaudioanalysis: An open-source python library for audio signal analysis. PloS one, 10(12). 

Google. Speech-to-text. 

Agnes Grünerbl, Amir Muaremi, Venet Osmani, Gernot Bahle, Stefan Öhler, Gerhard Tröster, Oscar May- ora, Christian Haring, and Paul Lukowicz. 2014. Smartphone-based recognition of states and state changes in bipolar disorder patients. 19. 

Lili Guo, Longbiao Wang, Jianwu Dang, Yahui Fu, Jiax- ing Liu, and Shifei Ding. 2022. Emotion recognition with multimodal transformer fusion framework based on acoustic and lexical information. 
Qin Jin, Chengxin Li, Shizhe Chen, and Huimin Wu. 2015. Speech emotion recognition with acoustic and lexical features. 

Eesung Kim and Jong Won Shin. 2019. Dnn-based emotion recognition based on bottleneck acoustic features and lexical features. 

Madhavan Mani, David J Kavanagh, Leanne Hides, and Stoyan R Stoyanov. 2015. Review and evaluation of mindfulness-based iphone apps. 3. 

Charles R. Marmar, Adam D. Brown, Meng Qian, Eu- gene Laska, Carole Siegel, Meng Li, Duna Abu- Amara, Andreas Tsiartas, Colleen Richey, Jennifer Smith, Bruce Knoth, and Dimitra Vergyri. 2019. Speech-based markers for posttraumatic stress disor- der in us veterans. 36. 

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representa- tions in vector space. 

S. Newman and V. G. Mather. 1938. Analysis of spoken language of patients with affective disorders. 94. 

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duch- esnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830. 

T. S. Polzin and A. Waibel. 2000. Emotion-sensitive human- computer interfaces. in Proc. ISCA Work- shop on Speech and Emotion. 

Darius A Rohani, Nanna Tuxen, Andrea Quemada Lopategui, Maria Faurholt-Jepsen, Lars V Kessing, and Jakob E Bardram. 2019. Benefits of using activ- ity recommender technology for self-management of depressive symptoms. volume 4. Proceedings of 13th EAI International Conference on Pervasive Comput- ing Technologies for Health Care. 

Peter J. Rousseeuw. 1987. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20:53–65. 

Gerard Salton and Michael J. McGill. 1986. Introduc- tion to Modern Information Retrieval. McGraw-Hill, Inc., New York, NY, USA. 

B. Schuller, R. Mullre, M. Lang, and G. Rigol. 2005. Speaker independent emotion recognition by early fusion of acoustic and linguistic features within en- sembles. in Proc. 9th Eurospeech - Interspeech. 

John Torous, Steven Richard Chan, Shih Yee-Marie Tan, Jacob Behrens, Ian Mathew, Erich J Conrad, Ladson Hinton, Peter Yellowlees, and Matcheri Keshavan. 2014. Patient smartphone ownership and interest in mobile apps to monitor symptoms of mental health conditions: A survey in four geographically distinct psychiatric clinics. 1. 

Felix Weninger, Florian Eyben, Björn Schuller, Mar- cello Mortillaro, and Klaus Scherer. 2013. On the acoustics of emotion in audio: What speech, music, and sound have in common. Frontiers in Psychology, 4. 

X. Zhe and A.C. Boucouvalas. 2002. Text-to-emotion engine for real time internet communication. in Proc. the Int. Symposium on Communication Systems, Net- works, and DSPs. 
