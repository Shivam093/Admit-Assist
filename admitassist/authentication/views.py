from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.db import connection
import re
from .models import Signup


# Create your views here.
def home(request):
    return render(request,"authentication/index3.html")

def result(request):
    major=int(request.GET.get("profile_evaluator[postgrad_major_id]"))
    
    # Import necessary libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    if major == 7:
        data = pd.read_csv('cs_dataset.csv')
        l1=['Arizona State University', 'Arkansas State University', 'Arkansas State University, Jonesboro', 'Boston University', 'Brown University', 'California State University, Chico', 'California State University, East Bay', 'California State University, Fresno', 'California State University, Long Beach', 'California State University, Los Angeles', 'California State University, Northridge', 'Carnegie Mellon University', 'Central Michigan University', 'Clemson University', 'Colorado State University', 'Colorado State University, Fort Collins', 'Columbia University', 'Concordia University, Montreal', 'Cornell University', 'Dartmouth College', 'DePaul University', 'Drexel University', 'Duke University', 'Florida State University', 'George Mason University', 'George Washington University', 'Georgia Institute of Technology', 'Georgia State University', 'Harvard University', 'Illinois Institute of Technology', 'Indiana University, Bloomington', 'Iowa State University', 'Johns Hopkins University', 'Kent State University', 'Lamar University', 'Massachusetts Institute of Technology', 'Michigan State University', 'Michigan Technological University', 'Mississippi State University', 'Missouri University of Science & Technology', 'Missouri University of Science and Technology', 'New Jersey Institute of Technology', 'New York Institute of Technology', 'New York Institute of Technology, Manhattan', 'New York Institute of Technology, Old Westbury', 'New York University', 'North Carolina State University', 'North Carolina State University, Raleigh', 'Northeastern University', 'Northern Arizona University', 'Northern Illinois University', 'Northwestern University', 'Old Dominion University', 'Oregon State University', 'Pace University', 'Pennsylvania State University', 'Portland State University', 'Purdue University, Northwest', 'Purdue University, West Lafayette', 'Rice University', 'Rochester Institute of Technology', 'Rutgers University, New Brunswick', 'SUNY, University at Buffalo', 'Sacred Heart University', 'San Diego State University', 'San Jose State University', 'Santa Clara University', 'Southern Illinois University Edwardsville', 'Southern Illinois University, Edwardsville', 'Southern Methodist University', 'Stanford University', 'State University of New York, Albany', 'State University of New York, Buffalo', 'State University of New York, Stony Brook', 'Stevens Institute of Technology', 'Syracuse University', 'Texas A&M University, College Station', 'Texas Tech University', 'University at Albany, SUNY', 'University of Alabama, Birmingham', 'University of Arizona', 'University of Bridgeport', 'University of California, Berkeley', 'University of California, Davis', 'University of California, Irvine', 'University of California, Los Angeles', 'University of California, Riverside', 'University of California, San Diego', 'University of California, Santa Barbara', 'University of California, Santa Cruz', 'University of Central Florida', 'University of Central Missouri', 'University of Central Oklahoma', 'University of Chicago', 'University of Cincinnati', 'University of Colorado, Boulder', 'University of Colorado, Denver', 'University of Connecticut', 'University of Florida', 'University of Florida, Gainesville', 'University of Houston', 'University of Houston, Clear Lake', 'University of Illinois, Chicago', 'University of Illinois, Springfield', 'University of Illinois, Urbana-Champaign', 'University of Maryland, Baltimore County', 'University of Maryland, College Park', 'University of Massachusetts, Amherst', 'University of Massachusetts, Lowell', 'University of Michigan, Ann Arbor', 'University of Minnesota, Twin Cities', 'University of Missouri, Columbia', 'University of Missouri, Kansas City', 'University of Nebraska, Omaha', 'University of New Hampshire', 'University of New Haven', 'University of North Carolina, Chapel Hill', 'University of North Carolina, Charlotte', 'University of North Texas', 'University of Pittsburgh', 'University of Pittsburgh, Pittsburgh Campus', 'University of Rochester', 'University of San Francisco', 'University of South Florida', 'University of Southern California', 'University of Tennessee, Knoxville', 'University of Texas, Arlington', 'University of Texas, Austin', 'University of Texas, Dallas', 'University of Utah', 'University of Virginia', 'University of Washington', 'University of Wisconsin, Madison', 'University of Wisconsin, Milwaukee', 'Virginia Tech University', 'Washington University in St. Louis', 'Western Illinois University', 'Wichita State University', 'Worcester Polytechnic Institute', 'Wright State University', 'Yale University']

    elif major == 20:
        data = pd.read_csv('electrical_dataset.csv')
        l1=['Arizona State University', 'Boston University', 'California State Polytechnic University, Pomona', 'California State University, Fresno', 'California State University, Fullerton', 'California State University, Long Beach', 'California State University, Los Angeles', 'California State University, Northridge', 'Case Western Reserve University', 'Clemson University', 'Cleveland State University', 'Colorado State University', 'Colorado State University, Fort Collins', 'Columbia University', 'Drexel University', 'Florida State University', 'George Washington University', 'Illinois Institute of Technology', 'Iowa State University', 'Michigan State University', 'Michigan Technological University', 'Missouri University of Science and Technology', 'New Jersey Institute of Technology', 'New York University', 'North Carolina State University', 'Northern Illinois University', 'Northwestern University', 'Pennsylvania State University', 'Rochester Institute of Technology', 'SUNY, University at Buffalo', 'San Diego State University', 'San Jose State University', 'Southern Illinois University, Edwardsville', 'Southern Methodist University', 'Stanford University', 'State University of New York, Buffalo', 'Stevens Institute of Technology', 'Syracuse University', 'Texas A&M University, College Station', 'Texas Tech University', 'University of Bridgeport', 'University of California, Berkeley', 'University of California, Irvine', 'University of California, Los Angeles', 'University of California, Riverside', 'University of Central Florida', 'University of Cincinnati', 'University of Colorado, Boulder', 'University of Dayton', 'University of Houston', 'University of Illinois, Chicago', 'University of Maryland, Baltimore', 'University of Massachusetts, Dartmouth', 'University of Massachusetts, Lowell', 'University of Michigan, Dearborn', 'University of Minnesota, Twin Cities', 'University of Missouri, Kansas City', 'University of New Haven', 'University of North Carolina, Charlotte', 'University of North Texas', 'University of Pennsylvania', 'University of Pittsburgh', 'University of South Florida', 'University of Southern California', 'University of Tennessee, Knoxville', 'University of Texas, Arlington', 'University of Texas, Dallas', 'University of Virginia', 'University of Washington', 'University of Wisconsin, Madison', 'Virginia Tech University', 'Wright State University']

    # Preprocess data
    data = data.dropna() # Remove missing values
    data = pd.get_dummies(data, columns=['Universities']) # One-hot encode university names
    X = data.drop(['Status', 'Target Major'], axis=1) # Features

    y = data['Status'] # Target variable
    #
    #X = (X - X.mean()) / X.std() # Scale features
    # Split dataset into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
# Make predictions on testing set
    y_pred = model.predict(X_test)
    '''
    model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    model.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(X_test)
    '''
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy for logistic:', accuracy)
    flag=0
    var1 = request.GET.get('profile_evaluator[gre_quant]')
    var2 = request.GET.get('profile_evaluator[gre_verbal]')

    if var1 is not None and var2 is not None:
        var1 = float(var1)
        var2 = float(var2)
        gre = var1 + var2
    var4=float(request.GET['profile_evaluator[gpa]'])
    if var4 >10:
        gpa= var4/10
    else:
        gpa=var4
    
    paper=float(request.GET['research'])
    work=float(request.GET['work-exp'])

    var3=(request.GET['profile_evaluator[english_score]'])
    if var3 == "toefl":
        flag=1
    elif var3 == "ielts":
        flag=2
    else:
        flag=0
    if flag==1:
        eng_score=float(request.GET['profile_evaluator[toefl]'])
        if (eng_score>=118) & (eng_score<=120):
            eng_score=9
        elif (eng_score>=115) & (eng_score<=117):
            eng_score=8.5
        elif (eng_score>=110) & (eng_score<=114):
            eng_score=8
        elif(eng_score>=102) & (eng_score<=109):
            eng_score=7.5
        elif(eng_score>=94) &(eng_score<=101):
            eng_score=7
        elif(eng_score>=79) & (eng_score<=93):
            eng_score=6.5
        elif(eng_score>=60) & (eng_score<=78):
            eng_score=6
        elif(eng_score>=46) & (eng_score<=59):
            eng_score=5.5
        elif(eng_score>=35) & (eng_score<=45):
            eng_score=5
        elif(eng_score>=32) & (eng_score<=34):
            eng_score=4.5
        else:
            eng_score=0
        
    
    elif flag==2:
        eng_score=float(request.GET['profile_evaluator[ielts]'])
    else:
        eng_score=0
    
    l2=[]
    l2.append(gre)
    l2.append(gpa)
    l2.append(paper)
    l2.append(work)
    l2.append(eng_score)
    temp=l2
    imp=(request.GET['profile_evaluator[dream_school_id]'])
    print(imp)
    for i in range(len(l1)):
        if imp != l1[i]:
            l2.append(0)
        else:
            l2.append(1)
    
    print(l2)
    a=np.array(l2).reshape(1,-1)
    a=model.predict_proba(a)
    print(a)
    a=a[0][0]
    a=a*100
    a=round(a,2)
    
    a=np.array2string(a)
    chance="The chance of getting an admit is :" + a+"%"

    safe=["Arkansas State University", "Illinois Institute of Technology", "Stevens Institute of Technology", "Rochester Institute of Technology", "Florida State University", "University of Massachusetts, Lowell", "New Jersey Institute of Technology", "North Carolina State University, Raleigh", "Southern Illinois University Edwardsville"]
    moderate=["Northeastern University","California State University, Long Beach", "University of Texas, Dallas", "University of Illinois, Chicago", "Arizona State University", "University of California, San Diego", "SUNY, University at Buffalo", "Rutgers University, New Brunswick", "Georgia State University", "Colorado State University"]
    ambitious=["New York University", "University of California, Berkeley", "University of California, Los Angeles", "University of Illinois, Urbana-Champaign", "University of Southern California", "Carnegie Mellon University", "Cornell University", "Yale University", "Northwestern University", "Johns Hopkins University"]

    import random

    # Generate 10 unique random numbers within a range
    num_list = random.sample(range(0, 8), 3)
    #l1=['Arizona State University', 'Arizona State University, Tempe', 'Arkansas State University', 'Arkansas State University, Jonesboro', 'Boston University', 'Brown University', 'California State University, Chico', 'California State University, East Bay', 'California State University, Fresno', 'California State University, Long Beach', 'California State University, Los Angeles', 'California State University, Northridge', 'Carnegie Mellon University', 'Central Michigan University', 'Clemson University', 'Colorado State University', 'Colorado State University, Fort Collins', 'Columbia University', 'Concordia University, Montreal', 'Cornell University', 'Dartmouth College', 'DePaul University', 'Drexel University', 'Duke University', 'Florida State University', 'George Mason University', 'George Washington University', 'Georgia Institute of Technology', 'Georgia State University', 'Harvard University', 'Illinois Institute of Technology', 'Indiana University, Bloomington', 'Iowa State University', 'Johns Hopkins University', 'Kent State University', 'Lamar University', 'Massachusetts Institute of Technology', 'Michigan State University', 'Michigan Technological University', 'Mississippi State University', 'Missouri University of Science & Technology', 'Missouri University of Science and Technology', 'New Jersey Institute of Technology', 'New York Institute of Technology', 'New York Institute of Technology, Manhattan', 'New York Institute of Technology, Old Westbury', 'New York University', 'North Carolina State University', 'North Carolina State University, Raleigh', 'Northeastern University', 'Northern Arizona University', 'Northern Illinois University', 'Northwestern University', 'Old Dominion University', 'Oregon State University', 'Pace University', 'Pennsylvania State University', 'Portland State University', 'Purdue University, Northwest', 'Purdue University, West Lafayette', 'Rice University', 'Rochester Institute of Technology', 'Rutgers University, New Brunswick', 'SUNY, University at Buffalo', 'Sacred Heart University', 'San Diego State University', 'San Jose State University', 'Santa Clara University', 'Southern Illinois University Edwardsville', 'Southern Illinois University, Edwardsville', 'Southern Methodist University', 'Stanford University', 'State University of New York, Albany', 'State University of New York, Buffalo', 'State University of New York, Stony Brook', 'Stevens Institute of Technology', 'Syracuse University', 'Texas A&M University, College Station', 'Texas Tech University', 'University at Albany, SUNY', 'University of Alabama, Birmingham', 'University of Arizona', 'University of Bridgeport', 'University of California, Berkeley', 'University of California, Davis', 'University of California, Irvine', 'University of California, Los Angeles', 'University of California, Riverside', 'University of California, San Diego', 'University of California, Santa Barbara', 'University of California, Santa Cruz', 'University of Central Florida', 'University of Central Missouri', 'University of Central Oklahoma', 'University of Chicago', 'University of Cincinnati', 'University of Colorado, Boulder', 'University of Colorado, Denver', 'University of Connecticut', 'University of Florida', 'University of Florida, Gainesville', 'University of Houston', 'University of Houston, Clear Lake', 'University of Illinois, Chicago', 'University of Illinois, Springfield', 'University of Illinois, Urbana-Champaign', 'University of Maryland, Baltimore County', 'University of Maryland, College Park', 'University of Massachusetts, Amherst', 'University of Massachusetts, Lowell', 'University of Michigan, Ann Arbor', 'University of Minnesota, Twin Cities', 'University of Missouri, Columbia', 'University of Missouri, Kansas City', 'University of Nebraska, Omaha', 'University of New Hampshire', 'University of New Haven', 'University of North Carolina, Chapel Hill', 'University of North Carolina, Charlotte', 'University of North Texas', 'University of Pittsburgh', 'University of Pittsburgh, Pittsburgh Campus', 'University of Rochester', 'University of San Francisco', 'University of South Florida', 'University of Southern California', 'University of Tennessee, Knoxville', 'University of Texas, Arlington', 'University of Texas, Austin', 'University of Texas, Dallas', 'University of Utah', 'University of Virginia', 'University of Washington', 'University of Wisconsin, Madison', 'University of Wisconsin, Milwaukee', 'Virginia Tech University', 'Washington University in St. Louis', 'Western Illinois University', 'Wichita State University', 'Worcester Polytechnic Institute', 'Wright State University', 'Yale University']
    safe1=[]
    for i in range(3):
    
        print(num_list)
        imp=safe[num_list[i]]
        print(imp)
        l3=[]
        l3.append(gre)
        l3.append(gpa)
        l3.append(paper)
        l3.append(work)
        l3.append(eng_score)
        print(l3)
        for i in range(len(l1)):
            if imp != l1[i]:
                l3.append(0)
            else:
                l3.append(1)
        print(l3)
        a=np.array(l3).reshape(1,-1)
        a=model.predict_proba(a)
        print(a)
        a=a[0][0]
        a=a*100
        a=round(a,2)
    
        a=np.array2string(a)
        b= imp +" is: " + a+"%"
        safe1.append(b)
    print(safe1)

    num_list = random.sample(range(0, 9), 3)
    #l1=['Arizona State University', 'Arizona State University, Tempe', 'Arkansas State University', 'Arkansas State University, Jonesboro', 'Boston University', 'Brown University', 'California State University, Chico', 'California State University, East Bay', 'California State University, Fresno', 'California State University, Long Beach', 'California State University, Los Angeles', 'California State University, Northridge', 'Carnegie Mellon University', 'Central Michigan University', 'Clemson University', 'Colorado State University', 'Colorado State University, Fort Collins', 'Columbia University', 'Concordia University, Montreal', 'Cornell University', 'Dartmouth College', 'DePaul University', 'Drexel University', 'Duke University', 'Florida State University', 'George Mason University', 'George Washington University', 'Georgia Institute of Technology', 'Georgia State University', 'Harvard University', 'Illinois Institute of Technology', 'Indiana University, Bloomington', 'Iowa State University', 'Johns Hopkins University', 'Kent State University', 'Lamar University', 'Massachusetts Institute of Technology', 'Michigan State University', 'Michigan Technological University', 'Mississippi State University', 'Missouri University of Science & Technology', 'Missouri University of Science and Technology', 'New Jersey Institute of Technology', 'New York Institute of Technology', 'New York Institute of Technology, Manhattan', 'New York Institute of Technology, Old Westbury', 'New York University', 'North Carolina State University', 'North Carolina State University, Raleigh', 'Northeastern University', 'Northern Arizona University', 'Northern Illinois University', 'Northwestern University', 'Old Dominion University', 'Oregon State University', 'Pace University', 'Pennsylvania State University', 'Portland State University', 'Purdue University, Northwest', 'Purdue University, West Lafayette', 'Rice University', 'Rochester Institute of Technology', 'Rutgers University, New Brunswick', 'SUNY, University at Buffalo', 'Sacred Heart University', 'San Diego State University', 'San Jose State University', 'Santa Clara University', 'Southern Illinois University Edwardsville', 'Southern Illinois University, Edwardsville', 'Southern Methodist University', 'Stanford University', 'State University of New York, Albany', 'State University of New York, Buffalo', 'State University of New York, Stony Brook', 'Stevens Institute of Technology', 'Syracuse University', 'Texas A&M University, College Station', 'Texas Tech University', 'University at Albany, SUNY', 'University of Alabama, Birmingham', 'University of Arizona', 'University of Bridgeport', 'University of California, Berkeley', 'University of California, Davis', 'University of California, Irvine', 'University of California, Los Angeles', 'University of California, Riverside', 'University of California, San Diego', 'University of California, Santa Barbara', 'University of California, Santa Cruz', 'University of Central Florida', 'University of Central Missouri', 'University of Central Oklahoma', 'University of Chicago', 'University of Cincinnati', 'University of Colorado, Boulder', 'University of Colorado, Denver', 'University of Connecticut', 'University of Florida', 'University of Florida, Gainesville', 'University of Houston', 'University of Houston, Clear Lake', 'University of Illinois, Chicago', 'University of Illinois, Springfield', 'University of Illinois, Urbana-Champaign', 'University of Maryland, Baltimore County', 'University of Maryland, College Park', 'University of Massachusetts, Amherst', 'University of Massachusetts, Lowell', 'University of Michigan, Ann Arbor', 'University of Minnesota, Twin Cities', 'University of Missouri, Columbia', 'University of Missouri, Kansas City', 'University of Nebraska, Omaha', 'University of New Hampshire', 'University of New Haven', 'University of North Carolina, Chapel Hill', 'University of North Carolina, Charlotte', 'University of North Texas', 'University of Pittsburgh', 'University of Pittsburgh, Pittsburgh Campus', 'University of Rochester', 'University of San Francisco', 'University of South Florida', 'University of Southern California', 'University of Tennessee, Knoxville', 'University of Texas, Arlington', 'University of Texas, Austin', 'University of Texas, Dallas', 'University of Utah', 'University of Virginia', 'University of Washington', 'University of Wisconsin, Madison', 'University of Wisconsin, Milwaukee', 'Virginia Tech University', 'Washington University in St. Louis', 'Western Illinois University', 'Wichita State University', 'Worcester Polytechnic Institute', 'Wright State University', 'Yale University']
    moderate1=[]
    for i in range(3):
    
        print(num_list)
        imp=moderate[num_list[i]]
        print(imp)
        l4=[]
        l4.append(gre)
        l4.append(gpa)
        l4.append(paper)
        l4.append(work)
        l4.append(eng_score)
        print(l4)
        for i in range(len(l1)):
            if imp != l1[i]:
                l4.append(0)
            else:
                l4.append(1)
        print(l4)
        a=np.array(l4).reshape(1,-1)
        a=model.predict_proba(a)
        print(a)
        a=a[0][0]
        a=a*100
        a=round(a,2)
    
        a=np.array2string(a)
        b= imp +" is: " + a+"%"
        moderate1.append(b)
    print(moderate1)

    num_list = random.sample(range(0, 8), 3)
    #l1=['Arizona State University', 'Arizona State University, Tempe', 'Arkansas State University', 'Arkansas State University, Jonesboro', 'Boston University', 'Brown University', 'California State University, Chico', 'California State University, East Bay', 'California State University, Fresno', 'California State University, Long Beach', 'California State University, Los Angeles', 'California State University, Northridge', 'Carnegie Mellon University', 'Central Michigan University', 'Clemson University', 'Colorado State University', 'Colorado State University, Fort Collins', 'Columbia University', 'Concordia University, Montreal', 'Cornell University', 'Dartmouth College', 'DePaul University', 'Drexel University', 'Duke University', 'Florida State University', 'George Mason University', 'George Washington University', 'Georgia Institute of Technology', 'Georgia State University', 'Harvard University', 'Illinois Institute of Technology', 'Indiana University, Bloomington', 'Iowa State University', 'Johns Hopkins University', 'Kent State University', 'Lamar University', 'Massachusetts Institute of Technology', 'Michigan State University', 'Michigan Technological University', 'Mississippi State University', 'Missouri University of Science & Technology', 'Missouri University of Science and Technology', 'New Jersey Institute of Technology', 'New York Institute of Technology', 'New York Institute of Technology, Manhattan', 'New York Institute of Technology, Old Westbury', 'New York University', 'North Carolina State University', 'North Carolina State University, Raleigh', 'Northeastern University', 'Northern Arizona University', 'Northern Illinois University', 'Northwestern University', 'Old Dominion University', 'Oregon State University', 'Pace University', 'Pennsylvania State University', 'Portland State University', 'Purdue University, Northwest', 'Purdue University, West Lafayette', 'Rice University', 'Rochester Institute of Technology', 'Rutgers University, New Brunswick', 'SUNY, University at Buffalo', 'Sacred Heart University', 'San Diego State University', 'San Jose State University', 'Santa Clara University', 'Southern Illinois University Edwardsville', 'Southern Illinois University, Edwardsville', 'Southern Methodist University', 'Stanford University', 'State University of New York, Albany', 'State University of New York, Buffalo', 'State University of New York, Stony Brook', 'Stevens Institute of Technology', 'Syracuse University', 'Texas A&M University, College Station', 'Texas Tech University', 'University at Albany, SUNY', 'University of Alabama, Birmingham', 'University of Arizona', 'University of Bridgeport', 'University of California, Berkeley', 'University of California, Davis', 'University of California, Irvine', 'University of California, Los Angeles', 'University of California, Riverside', 'University of California, San Diego', 'University of California, Santa Barbara', 'University of California, Santa Cruz', 'University of Central Florida', 'University of Central Missouri', 'University of Central Oklahoma', 'University of Chicago', 'University of Cincinnati', 'University of Colorado, Boulder', 'University of Colorado, Denver', 'University of Connecticut', 'University of Florida', 'University of Florida, Gainesville', 'University of Houston', 'University of Houston, Clear Lake', 'University of Illinois, Chicago', 'University of Illinois, Springfield', 'University of Illinois, Urbana-Champaign', 'University of Maryland, Baltimore County', 'University of Maryland, College Park', 'University of Massachusetts, Amherst', 'University of Massachusetts, Lowell', 'University of Michigan, Ann Arbor', 'University of Minnesota, Twin Cities', 'University of Missouri, Columbia', 'University of Missouri, Kansas City', 'University of Nebraska, Omaha', 'University of New Hampshire', 'University of New Haven', 'University of North Carolina, Chapel Hill', 'University of North Carolina, Charlotte', 'University of North Texas', 'University of Pittsburgh', 'University of Pittsburgh, Pittsburgh Campus', 'University of Rochester', 'University of San Francisco', 'University of South Florida', 'University of Southern California', 'University of Tennessee, Knoxville', 'University of Texas, Arlington', 'University of Texas, Austin', 'University of Texas, Dallas', 'University of Utah', 'University of Virginia', 'University of Washington', 'University of Wisconsin, Madison', 'University of Wisconsin, Milwaukee', 'Virginia Tech University', 'Washington University in St. Louis', 'Western Illinois University', 'Wichita State University', 'Worcester Polytechnic Institute', 'Wright State University', 'Yale University']
    ambitious1=[]
    for i in range(3):
    
        print(num_list)
        imp=ambitious[num_list[i]]
        print(imp)
        l5=[]
        l5.append(gre)
        l5.append(gpa)
        l5.append(paper)
        l5.append(work)
        l5.append(eng_score)
        print(l5)
        for i in range(len(l1)):
            if imp != l1[i]:
                l5.append(0)
            else:
                l5.append(1)
        print(l5)
        a=np.array(l5).reshape(1,-1)
        a=model.predict_proba(a)
        print(a)
        a=a[0][0]
        a=a*100
        a=round(a,2)
    
        a=np.array2string(a)
        b= imp +" is: " + a+"%"
        ambitious1.append(b)
    print(ambitious1)


    return render(request, "authentication/index3.html", {"result2":chance, "safe1":safe1[0], "safe2":safe1[1], "safe3":safe1[2], "moderate1":moderate1[0], "moderate2":moderate1[1], "moderate3":moderate1[2], "ambitious1":ambitious1[0], "ambitious2":ambitious1[1], "ambitious3":ambitious1[2]})

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['pass1']
        confirm_password = request.POST['pass2']
        if not username:
            messages.error(request, 'Username is required')
        elif len(username) < 4:
            messages.error(request, 'Username should be at least 4 characters long')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username is already taken')
        if not email:
            messages.error(request, 'Email is required')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email is already taken')
        if not password:
            messages.error(request, 'Password is required')
        elif len(password) < 8:
            messages.error(request, 'Password should be at least 8 characters long')
        elif password != confirm_password:
            messages.error(request, 'Passwords do not match')
        if not messages.get_messages(request):
            signup = Signup(username=username, email=email, password=password)
            signup.save()
            messages.success(request, "You have successfully signed up!")
            return redirect('home')
            
    return render(request, "authentication/signup.html")
    

def signin(request):
    if request.method == 'POST':
        user = request.POST['user']
        pass3 = request.POST['pass3']
        
        user = authenticate(username=username, password=pass1)
        if user is not None:
            login(request, user)
            username = user.username
            messages.success(request, "Logged In Sucessfully!!")
            return render(request, "authentication/index3.html",{"username":username})
        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('signup')

    