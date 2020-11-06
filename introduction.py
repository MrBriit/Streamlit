import streamlit as st


#Set title

st.title("Our First Streamlit App")

from PIL import Image

st.subheader('Total Data Science')

image=Image.open("tdslogo.png")
st.image(image,use_column_width=True)

st.write("writing a text here")

st.markdown("this is a markdown cell")

st.success("Congrat you run the App successfully")
st.info("this is an information for you")

st.warning("Be cautious")

st.error("Oops you run into an error, you need to rerun your app again or unstall and install your App again")

st.help(range)


import numpy as np 
import pandas as pd 

dataframe=np.random.rand(10,20)

st.dataframe(dataframe)


st.text("---"*100)

df=pd.DataFrame(np.random.rand(10,20), columns=('col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

st.text("---"*100)

#Display chart

chart_data=pd.DataFrame(np.random.randn(20,3), columns=['a','b','c'])

st.line_chart(chart_data)


st.text("---"*100)


st.area_chart(chart_data)



chart_data=pd.DataFrame(np.random.randn(50,3), columns=['a','b','c'])

st.bar_chart(chart_data)


import matplotlib.pyplot as plt 

arr=np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)

st.pyplot()



st.text("---"*100)


import plotly
import plotly.figure_factory as ff 


#Adding distplot

x1=np.random.randn(200)-2
x2=np.random.randn(200)
x3=np.random.randn(200)-2

hist_data=[x1,x2,x3]
group_labels=['Group1','Group2','Group3']

fig=ff.create_distplot(hist_data,group_labels,bin_size=[.2,.25,.5])

st.plotly_chart(fig,use_container_width=True)



st.text("---"*100)


df=pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4], columns=['lat','lon'])

st.map(df)




st.text("---"*100)

#creating buttons 


if st.button("Say hello"):
	st.write("hello is here")
else:
	st.write("why are you here")


st.text("---"*100)


genre=st.radio("What is your favourite genre?", ('Commedy','Drama','Documentary'))

if genre=='Commedy':
	st.write("Oh you like Commedy")
elif genre=='Drama':
	st.write("Yeah Drama is cool")
else:
	st.write(" i see!!")


st.text("---"*100)

#Select button

option=st.selectbox("How was your night?",('Fantastic','Awesome','So-so'))

st.write("Your said your night was:",option)



st.text("---"*100)


option=st.multiselect("How was your night, you can select multiple choice?",('Fantastic','Awesome','So-so'))

st.write("Your said your night was:",option)



st.text("---"*100)


age=st.slider('How old are you?',0,100,18)
st.write("Your age is : ",age)


values=st.slider('Select a range of values',0, 200,(15,80))

st.write('You selected a range between:', values)


number=st.number_input('Input number')
st.write('The number you inputed is:',number)


st.text("---"*100)
st.text("---"*100)



#File uploader

upload_file=st.file_uploader("Choose a csv file", type='csv')

if upload_file is not None:
	data=pd.read_csv(pload_file)
	st.write(data)
	st.success("successfully uploaded")
else:
	st.markdown("Please  upload a CSV file")

#Color picker 

color=st.sidebar.beta_color_picker("Pick your preferred color:",'#00f900')
st.write("This your color:",color)


#Side bar

st.text("---"*100)
st.text("---"*100)



add_sidebar=st.sidebar.selectbox("What is your favourite course?",('A course from TDS on building Data Web APP','Others', 'Am not sure'))


import time


my_bar=st.progress(0)
for percent_complete in range(100):
	time.sleep(0.1)
	my_bar.progress(percent_complete+1)

with st.spinner('wait for it...'):
	time.sleep(5)
st.success('successful')



st.balloons()





