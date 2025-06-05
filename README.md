# Emotion Classification on EEG Data
Authors: 110062130 沈錫賢, 110062144 王建智, 110062330 許昊彣

## Demo Video
 [![Watch the video](assets/Video.png)](https://www.youtube.com/watch?v=Mk1F4geuE28)


## Introduction

### Overview
- This project aims to develop an advanced EEG-based emotion classification model using the large-scale EEG dataset (FACED). By leveraging machine learning and signal processing techniques, we seek to significantly improve emotion recognition accuracy across 3 discrete emotional states (positive, neutral, negative).

### Data Description

- **Experimental Design/Paradigm:**

|   Item                    |       |  Content  |
|---------------------------|-------|-------------|
| Number of subjects        | 123     | 75 females, <br>mean age = 23.2 yrs|
| Emotion category          | 9       | Anger, Fear, Disgust, Sadness, Amusement, Inspiration, Joy, Tenderness, and Neutral|
| Number of video clips     | 28 |   |
| Self-reporting ratings     | 12 <br>(Scale from 0–7)  | Anger, Fear, Disgust, Sadness, Amusement, Inspiration, Joy, Tenderness, Valence, Arousal, Liking, and Familiarity |
| Recorded signals    | NeuSen.W32 | 32 channels, 250 Hz|

- **Procedure for Collecting Data:**

The figure below shows the experiment procedure of FACED dataset.
1. Each trial began with subjects focusing on a fixation cross, followed by a video. 
2. After each video, subjects were required to report on their subjective experiences during video-watching on 12 items.
3. Subjects provided ratings on a continuous scale of 0–7 for each item and then had at least 30 seconds of rest before starting the subsequent trial. 
4. To minimize the possible influence of alternating valence, video clips with the same valence (e.g., positive) were presented successively as a block of four trials. 
5. Between two blocks, subjects completed arithmetic problems to minimize the influence of previous emotional states on the subsequent block.

<img src="assets\dataset_procedure.png" width="800"/>
    
- **Hardware and Software Used:**

    - EEG Device: NeuSen.W32 (32 channels, 250 Hz)
    - Software: Psychophysics Toolbox 3.0 extensions in MATLAB

- **Data Size:**

    - 123 subjects
    - Formats: .bdf
    - Number of Channels:32 EEG channels
    - Sampling Rate: Original: 1000 or 250 Hz -> Downsampled: 250 Hz
    
- **Data Source:**

    - The dataset is publicly available from researchers at Tsinghua University, China.
    - We downloaded it from [[link]](https://www.synapse.org/Synapse:syn50614194/files/).
    
    
- **Quality Evaluation:** 
    -  ICLabel (ICA method: fastICA)
    <img src="assets\ICLabel_ICA.png" width="900"/>
    -  ICLabel (ICA method: infomax)
    <img src="assets\ICLabel_infomax.png" width="900"/>

    - In the **raw + fastICA** condition, **18 components** were classified as *brain*, which is suspiciously high. This likely reflects artifact contamination being misclassified as neural activity due to high noise levels and lack of preprocessing.
  
    - In contrast, the **ASR + Infomax** combination produced a more interpretable decomposition, with **8 brain components** and **8 eye blink components** clearly distinguished. This indicates that the data has been sufficiently denoised for ICA to separate neural and non-neural sources effectively, a sign of strong data integrity.

    - **Raw data** consistently resulted in a higher proportion of components labeled as *other* (e.g., **20 in raw + Infomax**), suggesting a lack of structure and signal quality, with noise overwhelming meaningful patterns.

    - With **preprocessing methods** such as filtering and ASR, ICLabel more confidently identifies known artifact categories (e.g., eye blinks, line noise), while reducing the ambiguous *other* category. This improves the interpretability and trustworthiness of the decomposed components.

    - Overall, **ASR-cleaned data combined with Infomax ICA** yielded the most reliable and credible results, as supported by ICLabel’s balanced classification of components into neural and artifact categories. Raw data, though sometimes appearing rich in "brain" components, is less trustworthy due to high noise levels and likely misclassifications.

    - In conclusion, these findings demonstrate the effectiveness of artifact removal and ICA configuration in enhancing data quality. The clear separation of brain and artifact components justifies the reliability of the EEG signals used in this project and supports the credibility of the subsequent classification results.

### Purpose
- In this project, we will conduct overall emotion classification research, concluding wave extraction, classification model training, and building a simple system that accepts the user’s brainwave as input and identifies the corresponding emotion as output. With emotion classification, doctors can analyze the patients’ emotions more objectively without looking at the hardly interpretable waveform but diagnosing by the result AI system predicts. In addition to medical applications, some daily life qualities can also be improved via emotion classification such as game experience improvement, marketing research and promotion, and many other fields.

## Model Framework

- The picture below is our BCI architecture

    <img src="assets\model_framework.png" width="900"/>
    
- Input/Output Mechanisms:
    - Input: Raw EEG data collected from FACED.
    - Output: Emotion classifications (Positive, Neutral, Negative).

- Signal Preprocessing Techniques:
    - Adjusting units: Different subjects may have different units (uV or V) of the recorded EEG signals; all were adjusted into uV.
    - Extract time window: Get the last 30 sec epoch for each video clip.
    - Down sample: 250 Hz
    - Band-Pass Filter: 0.05-47Hz
    - Handling bad channels: The detected bad channels were interpolated.
    - Independent Component Analysis (ICA): Remove artifacts, including eve movement, body movement.

- Feature Extraction:
    - The picture below is feature extraction architecture

        <img src="assets\feature_extraction.png" width="900"/>

    - Differential Entropy (DE): EEG signals may vary over time due to fluctuations in attention, which is the reason why we DE. DE will compute over successive time segments (1 sec), reducing the influence of time. It extracts the energy distribution across different frequency(figure below) bands from the EEG signals.

        | Frequency Band | Symbol | Range (Hz) |
        |----------------|--------|------------|
        | Delta          | δ      | 1–4 Hz     |
        | Theta          | θ      | 4–8 Hz     |
        | Alpha          | α      | 8–14 Hz    |
        | Beta           | β      | 14–30 Hz   |
        | Gamma          | γ      | 30–47 Hz   |

    
    - Decaying Incremental Normalization: Clips order for every subject is random, which will impact subjects’ emotions over time. Decaying incremental normalization reduces the impact of video order, allowing the normalization process to adapt to changes in the data distribution over time.
    The formula is as follows:
    $$
    \text{DIN}(i) = \frac{\sum_{j=1}^{N_i} w_{ij} \cdot x_{ij}}{\sqrt{\sum_{j=1}^{N_i} w_{ij}^2}}
    $$

    - Linear dynamic system (LDS): LDS smooths EEG data between time steps, reducing noise and capturing underlying patterns in the signal.

- Classification

  Since EEG data’s high complexity, we separate the 9 emotions into 3 categories Positive, Neutral, Negative.
    |      Positive                |   Neutral    |  Negative  |
    |---------------------------|-------|-------------|
    | Amusement, Inspiration, Joy, Tenderness        |  Neutral     | Anger, Fear, Disgust, Sadness |
    

  After transforming the data, we applied three different machine learning models to classify the emotions:
  
    * **SVM with linear & RBF kernel**
    	- SVM is a supervised ML algorithm mainly used for classification (and sometimes regression). It will find a hyperplane to separate the data into classes with largest margin.And we use 2 types of kernel in this research, linear and RBF(non-linear), and compare their result.

        | Parameter / Feature | Linear SVM| RBF SVM|
        |---------------------|-----------------------------|-------------------------------------------------------------------------------------------|
        | **C**               | 1e-5| 1e-5|
        | **Gamma**           | None| $\gamma = \frac{1}{n_{\text{features}} \times \text{Var}(X)}$|
        | **Multi-class**     | Uses one-vs-rest strategy.| Same as Linear SVM|

    * **XGboost**
        - XGboost is a ML algorithm based on gradient boosting. It is widely used for supervised learning tasks such as classification and regression.

        | Parameter          | Example Value(s)       |
        |--------------------|-----------------------|
        | **max_depth**          | 7                     |
        | **learning_rate**     | 0.1                   |
        | **n_estimators**       | 200                   |
        | **subsample**          | 0.8                   |
        | **colsample_bytree**   | 0.8                   |

 

## Validation
- 10-Fold Validation
    - 10-Fold Validation is a common form of cross-validation. We separate the dataset into 10 folders, then use 1 folder as testing set, others as training set. This process is repeated 10 times, with each folder used as testing set once. This validation method can estimate model performance more reliable and reduce the risk of overfitting.
    


## Usage
- Steps
    1. Download dataset from [[link]](https://www.synapse.org/Synapse:syn50614194/files/)
    2. Unzip the Data.zip
    3. Change the **"Data"** folder's name to **"data"**
    4. Change the **"data_dir"** global variable in *preprocess.ipynb* to your *data* folder's route
    5. We recommend running under Python version 3.11.6
    6. pip install -r requirements.txt
    7. Open and Run **preprocess.ipynb** for data preprocessing
    8. Open and Run **main.ipynb** for feature extraction, classification, result



## Results
- Confusion Matrix
    - By observing the 3 confusion matrices below, we can see the probability of being classified as neutral is much lower than positive and negative. This is because we re-classify the 9 emotions into 3 categories, which lead to the imbalance of data, and subsequently influence the model's performance. 

    1. linear SVC

    <img src="assets\lsvc1.png" width="400"/>


    2. RBF SVC

    <img src="assets\rsvc1.png" width="400"/>

    3. XGBoost

    <img src="assets\xg1.png" width="400"/>

    

- Accuracy
    - Linear SVM and XGboost both have the average accuracy close to 60%, while the RBF SVC archieve a relatively lower accuracy. On the other hand, we can find accuracy varies from different subjects, this is because self-rating is very subjective and which may not always reflect the real emotion state as the label.


    1. linear SVC

    <img src="assets\lsvc2.png" width="700"/>

    2. RBF SVC

    <img src="assets\rsvc2.png" width="700"/>

    3. XGBoost

    <img src="assets\xg2.png" width="700"/>
    <img src="assets\xgb_curve.png" width="700"/>

    
    
- Advantages and unique aspects 
    - The 2 figures below are from our reference study[[1]](#ref1), which classfied emotions into 9 category using a unique feature extraction method called CLISA（Contrastive Learning for Inter-Subject Alignment). The objective of CLISA was to reduce inter-subject differences, and it archieved accuracy of 42.4 ± 1.2%. In our research, we simplified the dataset; try different classification methods and sompare the result. We also observate the infuence of imbalance dataset on model performance.

    1. Accuracy of each subject from small to large

    <img src="assets\RWresult1.png" width="700"/>

    2. Confusion Matrix of 9 emotions

    <img src="assets\RWresult2.png" width="700"/>
    
    
    


## References

<a id="ref1"></a>[1] Chen, J., Wang, X., Huang, C. et al. A Large Finer-grained Affective Computing EEG Dataset. Sci Data 10, 740 (2023). https://doi.org/10.1038/s41597-023-02650-w