from flask import redirect,  url_for, render_template,request,Flask,Response,json,send_file
from match_pos import matcher
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy_try import spacy_match
import pandas as pd
import nltk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import random
import jinja2
#from pdf2docx import Converter
from bs4 import BeautifulSoup
from datetime import datetime
import openpyxl
from io import BytesIO
from excel_edit import edit_excel, read_excel

env = jinja2.Environment()
env.globals.update(zip=zip)


app = Flask(__name__)

df_for_excel = pd.DataFrame(columns=['Name','Match_Percentage'])
df_for_excel = pd.DataFrame(None)
#df_for_excel = df_for_excel.reset_index(drop=True, inplace=True)

count_excel_path = 'count.xlsx'

@app.route('/home/',methods=['GET','POST'])
def realhome():
	return render_template('tool_home.html')


@app.route('/jdres/',methods=['GET','POST'])
	#getting initial count of resumes checked
	initial_count = read_excel(count_excel_path,'Sheet1','A1')
	df_for_excel = pd.DataFrame(None)

	return render_template('index.html',initial_count=initial_count)

@app.route("/result", methods=['GET','POST'])

def result():
	if request.method == 'POST':
		#jd = request.files['jd']
		#resume = request.files['resume']
		if request.files['jd'].filename != "":
			jd = request.files['jd']
			if request.files['resume'].filename!="":
				resume = request.files.getlist('resume')
				print("this is the list:")
				name_list = []
				print(name_list)
				i = 0
				while i < len(resume):
					name_list.append(resume[i].filename.split(".")[0])
					i=i+1
				id_list = []
				percentage,word_list,common_list=matcher(resume,jd)				
				k=0
				while k < len(percentage):
					id_list.append(k)
					k=k+1
				zip_list = zip(percentage,word_list,common_list,name_list,id_list)
				
				return render_template("result.html",zip_list = zip_list)#percentage=percentage,word_list=word_list, common_list= common_list,count_skill=len(word_list),count_common=len(common_list))
			else:
				return render_template("notfound.html")
		else: 
			return render_template("notfound.html")

@app.route("/newresult",methods=['GET','POST'])

def newresult():
	if request.method == 'POST':
		#jd = request.files['jd']
		#resume = request.files['resume']
		if request.files['jd'].filename != "":
			jd = request.files['jd']
			if request.files['resume'].filename!="":
				resume = request.files.getlist('resume')
				check_type = request.form['role_type']
				print("this is the list:")

				name_list = []

				i = 0
				while i < len(resume):
					name_list.append(resume[i].filename.split(".")[0])
					i=i+1
				#counting how many resumes are being checked
				count_of_resume = len(name_list)
				print("the count of resume is:::")
				print(count_of_resume)
				#getting initial count of resumes checked
				initial_count = read_excel('count.xlsx','Sheet1','A1')
				print("the intial count value is::::")
				print(initial_count)
				#adding new resume count values
				new_count_value = initial_count+count_of_resume
				edit_excel('count.xlsx','Sheet1','A1',new_count_value)

				id_list = []
				percentage,word_list,common_list,label_x,resume_val,jd_val=spacy_match(resume,jd,check_type)				
				k=0
				while k < len(percentage):
					id_list.append(k)
					k=k+1
				zip_list_1 = zip(label_x,resume_val,jd_val)
				zip_list = zip(percentage,word_list,common_list,name_list,id_list)
				
				df_for_excel["Name"] = pd.Series(name_list)
				df_for_excel["Match_Percentage"] = pd.Series(percentage)
				
				
				return render_template("newresult.html",zip_list = zip_list,zip_list_1=zip_list_1)
			else:
				return render_template("notfound.html")
		else: 
			return render_template("notfound.html")

		
@app.route('/downloadresult/', methods=['GET', 'POST'])
def downloadresult():
	if request.method == 'POST':
		output = BytesIO()
		writer = pd.ExcelWriter(output, engine='xlsxwriter')
		df_for_excel.to_excel(writer,startrow = 0, merge_cells = False, sheet_name = "Sheet_1")
		
		workbook = writer.book
		worksheet = writer.sheets["Sheet_1"]
		writer.close()
		output.seek(0)
		
		return send_file(output, attachment_filename="output.xlsx", as_attachment=True)

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig



if __name__ == '__main__':
    app.debug = True
    app.run(host = '127.0.0.1', port= 5000)


