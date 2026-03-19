## Exno:1
### Data Cleaning Process
### AIM
To read the given data and perform data cleaning and save the cleaned data to a file.
### Explanation
##### Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.
### Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

### Coding and Output:
#### Data cleaning process:

```
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS.csv")
df
```
<img width="869" height="710" alt="363189923-93401c95-bb10-41fd-879e-4969a5f7ee4f" src="https://github.com/user-attachments/assets/ce2b7cfa-5c68-4ded-8acb-7b05e6964276" />


```
df.head()
```
<img width="851" height="198" alt="363190130-1a447fe0-a978-4c92-87b7-a98c6c1d7d84" src="https://github.com/user-attachments/assets/ae1fec85-8f60-479b-b4ce-360aec43edf3" />


```
df.tail(5)
```
<img width="857" height="201" alt="363190267-5cda549c-de93-45b7-ba82-b7f5f5e686af" src="https://github.com/user-attachments/assets/86a1fa3d-f521-47e2-8e77-733823a6cf25" />


```
df.isnull()
```
<img width="752" height="706" alt="363190427-77005ab8-affc-49a0-94bb-e85103cdba08" src="https://github.com/user-attachments/assets/33b72151-92c1-4a0d-ad34-4c92af8bf2e3" />


```
df.notnull()
```
<img width="649" height="683" alt="363190737-01585993-3e36-4f64-b635-bec23b09a13a" src="https://github.com/user-attachments/assets/4e525c73-54ce-40a8-8158-32f2f44db258" />


```
df.dropna(axis=0)
```
<img width="858" height="456" alt="363190930-2f67b824-36bd-4f8b-9b45-e101d99f273c" src="https://github.com/user-attachments/assets/ea92bae2-7928-4750-a9da-7bbc2b89d4c1" />


```
df.dropna(axis=1)
```
<img width="226" height="687" alt="363191042-6a6d8a81-5109-4079-aa55-9de0774f7a66" src="https://github.com/user-attachments/assets/7732cc98-208a-48dd-9231-4acb0e8f88f6" />


```
df.fillna(0)
```
<img width="827" height="694" alt="363191283-366417b2-0fd6-4a00-b2d8-99bdb7c62661" src="https://github.com/user-attachments/assets/4f1bf4f0-d6b8-4fb0-8755-89063cabe3f2" />


```
print(df.shape)
```

<img width="371" height="44" alt="363191408-bef54714-c7f9-48a6-89f7-71feb365d436" src="https://github.com/user-attachments/assets/692cd41c-24ba-4a63-beb9-3a1e8ff53cd9" />


### IQR:

```
import pandas as pd
import seaborn as sns
ir=pd.read_csv('/content/iris.csv')
ir
```
<img width="579" height="408" alt="363988707-41fc23ee-ac24-4c9d-b9cf-ce683d86d862" src="https://github.com/user-attachments/assets/1e56428a-03ec-40c2-9a9a-bea4a1d7f897" />


```
ir.describe()
```

<img width="507" height="283" alt="363192018-c0540501-73de-4ecc-9d1f-ead4961372cd" src="https://github.com/user-attachments/assets/1b9c9f38-007d-4a23-9eaf-ff0b62e157c5" />


```
sns.boxplot(x='sepal_width',data=ir)
```

<img width="560" height="457" alt="363192166-452705a2-1d41-45ed-9897-1fd921770faa" src="https://github.com/user-attachments/assets/03d866ce-b6c4-4edc-8518-bb2e70499ebe" />


```
c1=ir.sepal_width.quantile(0.25)
c3=ir.sepal_width.quantile(0.75)
iq=c3-c1
print(c3)
```

<img width="149" height="181" alt="363192422-0e6ce3cc-3d98-44bb-b68d-5fb8d1ac666b" src="https://github.com/user-attachments/assets/a355dbad-6a0c-49f2-ba03-b70512f58a69" />


```
rid=ir[((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
```
<img width="594" height="409" alt="363192614-6a133748-77c2-4dca-8f77-8b278b5d4505" src="https://github.com/user-attachments/assets/fc6f4f1b-9c5f-44e0-bb02-a641faa53d48" />


```
delid=ir[~((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
delid
```

<img width="566" height="462" alt="363192731-064fc55c-0136-4e32-af0f-f47b2d6e9f86" src="https://github.com/user-attachments/assets/63d4d4fd-55d1-45ea-8575-3d1b1c0d668e" />


```
sns.boxplot(x='sepal_width',data=delid)
```
<img width="566" height="462" alt="363192731-064fc55c-0136-4e32-af0f-f47b2d6e9f86" src="https://github.com/user-attachments/assets/a2297c38-3992-4618-a9d6-095123274462" />



### Z SQUARE

```
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
import scipy.stats as stats
dataset=pd.read_csv("/content/heights.csv")
dataset
```

<img width="211" height="471" alt="363193142-91aaebae-9883-4d8b-8973-ca9062d26978" src="https://github.com/user-attachments/assets/3d484065-e8c1-4a2f-808a-f46fd65f61ab" />


```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```

<img width="233" height="406" alt="363193945-8b3080a2-d1b8-4a38-98d1-1f141bdacbd4" src="https://github.com/user-attachments/assets/85607db7-e891-4513-bb72-0837607fb7d7" />


```
low = q1 - 1.5*iqr
low
```
<img width="102" height="505" alt="363194166-20e5a03a-23e3-448b-bbb7-11f5fce6bbad" src="https://github.com/user-attachments/assets/78d80088-d391-44ed-88c5-b6f0e6c78a64" />



```
high = q3 + 1.5*iqr
high
```

<img width="225" height="447" alt="363194365-b0dc52f3-9a89-4a0d-b598-9ece1cda054b" src="https://github.com/user-attachments/assets/ebcb00ea-2ea4-4512-80b8-d493ad7c6560" />


```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```

![image](https://github.com/user-attachments/assets/8b3080a2-d1b8-4a38-98d1-1f141bdacbd4)

```
z = np.abs(stats.zscore(df['height']))
z
```

![image](https://github.com/user-attachments/assets/20e5a03a-23e3-448b-bbb7-11f5fce6bbad)

```
 df1 = df[z<3]
 df1
```

![image](https://github.com/user-attachments/assets/b0dc52f3-9a89-4a0d-b598-9ece1cda054b)

### Result
Thus the Data Cleaning Process and Detecting and Removal of Outliers is executed successfully.     
