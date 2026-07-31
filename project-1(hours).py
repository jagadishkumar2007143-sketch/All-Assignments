import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = {
    "Study_hours" : [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5],
    "Exam_Score" : [5,10,15,20,25,30,35,40,45,50]
}

df = pd.DataFrame(data)

print(df)


