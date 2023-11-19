import json
import os
import pickle
from datetime import datetime
import warnings

#import openai
import pandas as pd
import requests
import streamlit as st


def register_user(uploaded_file, name, dob, email, password):
  # Check if email already exists in CSV
  existing_users = pd.read_csv('user_data.csv')
  existing_emails = existing_users['Email'].tolist()
  if email in existing_emails:
    st.error("Email address already registered.")
    return False

  # Validate input and save user information to CSV
  if not uploaded_file:
    st.error("Please upload a profile photo.")
    return False

  if not name or not dob or not email or not password:
    st.error("Please fill in all required fields.")
    return False

  if not is_valid_email(email):
    st.error("Please enter a valid email address.")
    return False

  if len(password) < 8:
    st.error("Password must be at least 8 characters long.")
    return False

  # Save uploaded photo to a permanent location
  photo_filename = f"{email}.jpg"
  with open(photo_filename, 'wb') as f:
    f.write(uploaded_file.read())

  # Save user information to CSV
  user_data = {
      'Photo': photo_filename,
      'Name': name,
      'DOB': dob,
      'Email': email,
      'Password': password
  }

  df = pd.DataFrame([user_data])
  df.to_csv('user_data.csv', mode='a', header=False, index=False)

  st.success('Registration successful!')
  return True


def verify_user(email, password):
  # Check if email exists in CSV
  existing_users = pd.read_csv('user_data.csv')
  existing_emails = existing_users['Email'].tolist()
  if email not in existing_emails:
    st.error("Incorrect email address.")
    return False

  # Verify password using data from CSV
  user_data = existing_users[existing_users['Email'] == email]
  if password != user_data['Password'].values[0]:
    st.error('Incorrect password.')
    return False

  # Set session variable to indicate successful login
  st.session_state['logged_in'] = True
  return True


def is_valid_email(email):
  regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]+$'
  return re.match(regex, email)


# Function to preprocess TSSM (Time Spent on Social Media)
def preprocess_TSSM(TSSM):
  min_TSSM = 0  # Replace with the minimum TSSM value in your dataset
  max_TSSM = 10  # Replace with the maximum TSSM value in your dataset
  normalized_TSSM = (TSSM - min_TSSM) / (max_TSSM - min_TSSM)
  return normalized_TSSM


# Function to calculate Age from Date of Birth (DOB)
def calculate_age_from_DOB(DOB):
  birth_date = datetime.strptime(DOB, '%Y-%m-%d')
  current_date = datetime.now()
  age = current_date.year - birth_date.year - (
      (current_date.month, current_date.day) <
      (birth_date.month, birth_date.day))
  return age


# Function to make prediction using XGBoost model
def make_prediction(TSSM, DOB):
  normalized_TSSM = preprocess_TSSM(TSSM)
  age = calculate_age_from_DOB(DOB)
  with open('xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)
  prediction = xgb_model.predict([[normalized_TSSM, age]])
  return prediction[0]


# Function to generate advice using GPT-3.5-turbo API


def generate_advice(prediction):
  # Open AI API key
  openai_api_key = "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"

  if prediction is not None:
    pass
  else:
    pass

  prompt = f"Provide advice for user to help them manage their mental health.Based ontheir social media usage, it is has been founnd they at this level of risk of depression ${prediction:.2f} for their monthly expenses. Please offer a one sentence advice of not more than 7 words on how they can relax their mind ."

  openai_endpoint = "https://api.openai.com/v1/chat/completions"
  headers = {
      "Content-Type": "application/json",
      "Authorization": "Bearer {}".format(openai_api_key)
  }

  data = {
      "model": "gpt-3.5-turbo",
      "messages": [{
          "role": "user",
          "content": prompt
      }]
  }

  response = requests.post(openai_endpoint, headers=headers, json=data)
  response_data = json.loads(response.text)

  advice = response_data["choices"][0]["message"]["content"]
  return advice


# Initialize session variable for login status
st.session_state['logged_in'] = False

# Load existing user data from CSV file
with warnings.catch_warnings(record=True):
  existing_users = pd.read_csv('user_data.csv')
  existing_emails = existing_users['Email'].tolist()

# Define the filename for the Users CSV
users_file = 'user_data.csv'

# Sidebar navigation
st.sidebar.title('HappyMe')
selected_page = st.sidebar.radio('Navigation', ['Signup', 'Signin', 'Profile'])

# Signup screen
if selected_page == 'Signup':
  st.title('Signup')

  # Upload profile photo
  uploaded_file = st.file_uploader('Select a profile photo')
  if uploaded_file is not None:
    # Save uploaded photo to a temporary location
    with open('temp_photo.jpg', 'wb') as f:
      f.write(uploaded_file.read())

  # Collect user information
  name = st.text_input('Name:')
  dob = st.date_input('Date of Birth:')
  email = st.text_input('Email:')
  password = st.text_input('Password:', type='password')

  # Submit signup form
  if st.button('Signup'):
    # Validate and save user registration
    if register_user(uploaded_file, name, dob, email, password):
      st.success('Registration successful!')
      st.session_state['logged_in'] = True
      selected_page = 'Profile'

# Signin screen
elif selected_page == 'Signin':
  st.title('Signin')

  email = st.text_input('Email:')
  password = st.text_input('Password:', type='password')

  # Submit signin form
  if st.button('Signin'):
    if verify_user(email, password):
      st.success('Sign-in successful!')
      selected_page = 'Profile'
    else:
      st.error('Incorrect credentials.')

# Profile screen
elif selected_page == 'Profile':
  st.title('Profile')

  # Display profile information for logged-in user
  if st.session_state.get('logged_in'):
    # Retrieve user information from CSV based on logged-in user's email
    user_info = existing_users[existing_users['Email'] ==
                               st.session_state['email']]
    user_photo = user_info['Photo'].values[0]
    user_name = user_info['Name'].values[0]
    user_email = user_info['Email'].values[0]

    # Display user's photo, name, and email
    st.image(user_photo)
    st.write(f"Name: {user_name}")
    st.write(f"Email: {user_email}")

  st.sidebar.title('HappyMe')
  selected_page_1 = st.sidebar.radio('Navigation',
                                     ['Profile', 'Dashboard', 'History'])
  #NEW SIDEBAR NAVIGATION
  # Define the filename for the history CSV
  history_file = 'History.csv'
  # Profile screen
  if selected_page_1 == 'Profile':

    # Display profile information for logged-in user
    if st.session_state.get('logged_in'):
      # Retrieve user information from CSV based on logged-in user's email
      user_info = existing_users[existing_users['Email'] ==
                                 st.session_state['email']]
      user_photo = user_info['Photo'].values[0]
      user_name = user_info['Name'].values[0]
      user_email = user_info['Email'].values[0]

      # Display user's photo, name, and email
      st.image(user_photo)
      st.write(f"Name: {user_name}")
      st.write(f"Email: {user_email}")

  # Dashboard screen
  elif selected_page_1 == 'Dashboard':
    st.title('Dashboard')
    TSSM = st.text_input('Enter Time Spent on Social Media:')

    if st.button('Make Prediction'):
      user_info = existing_users[existing_users['Email'] ==
                                 st.session_state['mail']]
      DOB = user_info['DOB'].values[0]
      prediction = make_prediction(float(TSSM), DOB)
      advice = generate_advice(prediction)

      st.write(f"Predicted Mental State Category: {prediction}")
      st.write(f"Advice: {advice}")

      # Prepare the data to be stored in the History CSV
      current_data = {
          'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
          'Mental_State_Category': prediction,
          'Advice': advice,
      }

      # Check if the History file already exists
      if not os.path.exists(history_file):
        # If the file doesn't exist, create a new CSV and save the current data
        history_df = pd.DataFrame([current_data])
        history_df.to_csv(history_file, index=False)
      else:
        # If the file exists, load the existing data, append the current data, and save it
        history_df = pd.read_csv(history_file)
        history_df = history_df.append(current_data, ignore_index=True)
        history_df.to_csv(history_file, index=False)

        st.write("Data saved to History file.")

  # History screen
  elif selected_page_1 == 'History':
    st.title('History')
    # Retrieve historical data from CSV file
    df = pd.read_csv(history_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Display historical data
    st.subheader('Historical Data')
    st.table(df)
  else:
    st.warning('Please sign in to access your profile.')
