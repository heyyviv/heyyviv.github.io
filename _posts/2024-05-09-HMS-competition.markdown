---
layout: post
title: "Detecting seizure from eeg and spectrogram part 1"
date: 2024-05-09 01:43:18 +0530
categories: Deep Learning,Kaggle
---

HMS DATA:

Overall, 124 raters scored 50,697 EEG segments from 2,711 distinct patients’ EEGs. Among the 124 raters, They
identified a subset of 20 physician experts with subspecialty training who each individually annotated ≥1000 EEG
segments. Among EEG segments annotated by these 20 experts, 9,129 segments received labels from at least 10
independent experts. Based on prior work suggesting that ~10 experts are required to achieve a stable group
consensus1, They designated these segments as having “high quality” labels, meaning that these samples received
enough expert labels so that the consensus label (IIIC pattern type with the most labels) and degree of agreement
amongst experts about the correct label can be assigned with high confidence. The remaining labeled data was
considered “lower quality” e.g., because these segments either had fewer labels or not all raters had fellowship
training. They therefore divided the 50,697 labeled EEG segments into a group of 9,129 segments with “high
quality” labels scored by 20 “top experts”; the remaining set of 30,628 segments had lower-quality labels
collectively received from 119 of the 124 raters. 

They expanded the high- and low-quality sets of EEG segments by adding additional segments belonging to the 
same stationary period of the EEG. This yielded 71,982 segments with high-quality labels from 1,522 patients,
and 111,095 segments with lower-quality labels from 1,950 patients. Nevertheless, in the final division of data
into training and test datasets, any given patient’s data is assigned exclusively to either the training or to the test
dataset. 

The justification for expanding labeled data is as follows:
Empirically, the EEG can be divided into a series of “stationary periods” (SP), within which the pattern of EEG activity is unchanging. These SP can be identified by detecting changepoints within the EEG power, as illustrated in Figure . 
The upper four subplots of Figure show spectrograms from four brain regions (LL = left lateral, RL = right lateral, LP = left para-central, and RP =right para-central). A seizure occurs near the center of the image. Below the spectrogram is shown the sum of
the total spectral power across all four regions, and the SP between changepoints (CPs, identified by a CP  detection algorithm). Raw EEG from three different segments within the SP (region within the central part of the
seizure, between the two central pink vertical lines) are shown below, indicated by their starting times (t1, tC, t2),
where “tC” represents the time at the center of SP shown. Each 10-sec EEG segment within the same SP
demonstrates clear seizure activity, like the central segment. This example illustrates the rationale for assigning
the same label to all EEG segments that fall within the same SP. In our approach, experts label the central EEG
segment, and the same label is then automatically assigned to the remaining segments within the SP, increasing
the number of labeled samples available for model training and evaluation.

![EEG](/assets/HMS1.jpg)

Not only was the test data filtered by having more than 9 votes, but also the data has been augmented by using the same labels but shifting the eeg data. This means that the given 106k training rows are highly redundant and can/ should be filtered.

{% highlight python %}
from mne.filter import filter_data, notch_filter
def eeg_16C(eeg_path,plot=False):
    data = pd.read_parquet(eeg_path).values
    sample = data.T[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]]\
             - data.T[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]]
    sample = notch_filter(sample.astype('float64'), 200, 60, n_jobs=32, verbose='ERROR')
    #200 is sampling rate ,quality factor=60
    #Removing power-line noise can be done with a Notch filter, directly on the Raw object
    sample = filter_data(sample.astype('float64'), 200, 0.5, 40, n_jobs=32, verbose='ERROR') 
    sample = np.clip(sample,-500,500)
    #Given an interval, values outside the interval are clipped to the interval edges
    sample = np.nan_to_num(sample, nan=0)
    
    if plot:
        plt.figure(figsize=(20,10))
        d=0
        for i in range(16):
            plt.plot(np.arange(10_000),sample[i,]+d)
            d+=np.max(sample[i,])
        plt.title("16 montage banana")
        plt.show()
                     
{% endhighlight %}
![Double banana](/assets/HMS2.jpg)
![16 montage banana plot](/assets/hms3.jpg)

preparing data
removing weak data and redundant data 

{% highlight python %}
train_csv = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
print(len(train_csv))
target =['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']
train_csv['list_sum'] = train_csv[target].sum(axis=1)
train_csv = train_csv[train_csv['list_sum']>9]
train_csv = train_csv.drop(columns = ['list_sum'])
print(len(train_csv))
train_df = train_csv.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
                                'spectrogram_id' : 'first',
                                'spectrogram_label_offset_seconds' : 'min'
                                })
train_df.columns = ['spectrogram_id','min']

aux = train_csv.groupby('eeg_id')[['spectrogram_label_offset_seconds']].agg({
    'spectrogram_label_offset_seconds' : 'max'
})
train_df['max']=aux
aux = train_csv.groupby('eeg_id')[['patient_id']].agg('first')
train_df['patient_id'] = aux
aux = train_csv.groupby('eeg_id')[target].agg('sum')
for label in target:
    train_df[label]=aux[label].values
y_data = train_df[target].values
y_data = y_data/y_data.sum(axis=1,keepdims=True)
train_df[target]=y_data
aux = train_csv.groupby('eeg_id')[['expert_consensus']].agg('first')
train_df['target']=aux
train = train_df.reset_index()
print(len(train))
train.head()
{% endhighlight %}


