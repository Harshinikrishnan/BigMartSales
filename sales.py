# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle 


loaded_model=pickle.load(open("C:\\Users\\User\\Desktop\\big mart sales\\trained_model.sav","rb"))

input_data = (156,	9.30,	0,	0.016047,	4,	249.8092	,9,	1999,	1,	0,	1	)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


