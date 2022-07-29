# tsar: Time Series Automated Re-labeling

Accurately detecting instances in datasets that have been mislabeled is a difficult problem with several imperfect solutions. Hand-reviewing labels is a reliable but expensive approach. Time series datasets present additional challenges because they are not as easily interpreted by reviewers. This paper introduces TSAR, as system for facilitating human review of a small portion of a dataset that it identifies as the most likely to be mislabeled. TSAR's use is demonstrated on real-world time series data.

## Use
TSAR provides to main functions: identifying a list of instances in a dataset which are the most likely to be mislabeled and generating interpretable visualizations for each of those instances. As the name indicated TSAR is intended to be used with time series data. This may include: accelerometers, gyroscopes, ECG, EEG, temperatures, flow rates, or many other sensors. TSAR employs a 1D CNN as a tool for identifying its candidate list and as such can be used to review any dataset with a fixed input vector size but your results may vary for data domains other than time series.

## Candidate List
A deep neural network is trained on an appropriate dataset and then predicts a label for each instance. The list of instances is sorted by the dot product of the predicted label and the assigned label. Predicted abels that are farther from the assigned labels will have a smaller value (an orthogonal one-hot encoded label will have a value of zero) and they go to the top of the list. TSAR will generate visualizations for some percentage (user input) of the full dataset (these are the candidate list). Values of 2% to 7% are appropriate in our experience. Less won't have much effect and more gets unweidly to review.

## Visualizations
TSAR uses several tricks to make your data more interpretable. The first is to generate rich feature sets using deep feature learners (that's where the CNN come in) and to use that feature set and tSNE to produce a scatter plot of some instances from your data (either all instances or only the two most likelty correct classes. The latter is less cluttered and is prefered by this team). Mislabeled point in this scatter plot will often fall into a miscolored region of the plot. The visualization also includes three waveforms. The first is the instance being reviewed. Second the nearest neighbor in the feature set with the SAME label, Third the nearest neighbor in the feature set with a DIFFERENT label.# tsar: Time Series Automated Re-labeling

This work is licensed under Creative Commons Attribution 4.0 International Public License. Please consult the license file for more information about permitted activities and requirements. 

## Cite

Atkinson, G., & Metsis, V. (2021, June). TSAR: a time series assisted relabeling tool for reducing label noise. In The *14th PErvasive Technologies Related to Assistive Environments Conference* (pp. 203-209).

<pre>
@inproceedings{atkinson2021tsar,
  title={TSAR: a time series assisted relabeling tool for reducing label noise},
  author={Atkinson, Gentry and Metsis, Vangelis},
  booktitle={The 14th PErvasive Technologies Related to Assistive Environments Conference},
  pages={203--209},
  year={2021}
}
</pre>
