# lightweightML

The main purpose of this project is the Classification of Body Postures and Movements for <b>(PUC-Rio)</b> Data Set.<br>
During the implementation process I try to make the project as generic as possible. <br>
Most of the codes are data pre-processing codes that can be used for other type of data sets.<br>
Machine learning algorithms used are k-NN and AdaBoost.<br>

<i>Steps of the program:</i><br>
1. Pre-process data sets<br>
&nbsp;&nbsp;a) Split data by class<br>
&nbsp;&nbsp;b) Split data to K-fold randomly<br>
2. Prepare data sets for learning process<br>
&nbsp;&nbsp;a) Normalize data sets<br>
&nbsp;&nbsp;b) Combine data sets into K-fold and use one of the fold for validation<br>
3. Training and validation<br>
&nbsp;&nbsp;a) Find the suitable parameters for the classifier(s)<br>
&nbsp;&nbsp;b) Log the result to file<br>
4. Classify input data with the resulting classifier(s)<br>
&nbsp;&nbsp;a) Find classifier with lowest error for each class (in the case of multi-class)<br>
&nbsp;&nbsp;b) Classify the data and write the result to file<br>
