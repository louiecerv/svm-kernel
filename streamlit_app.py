import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    st.title('Comparison of SVM Kernels')

    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 1    

    # Display the appropriate form based on the current form state
    if st.session_state["current_form"] == 1:
        display_form1()
    elif st.session_state["current_form"] == 2:
        display_form2()
    elif st.session_state["current_form"] == 3:
        display_form3()

    if "form2" not in st.session_state: 
        st.session_state["form2"] = []

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")
    form1.subheader('About the Classifier')

    text = """Louie F. Cervantes, M.Eng. \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    form1.text(text)

    form1.subheader('Description')
    form1.write('Support Vector Machines (SVM):')
    text = """Supervised learning algorithm: Used for both classification and regression.
        Linear decision boundary: In high-dimensional spaces, it uses the 
        kernel trick to create a non-linear decision boundary by implicitly 
        mapping data points to higher dimensions.
        Maximizing margin: Aims to find the hyperplane that separates classes 
        with the largest margin (distance between the hyperplane and the closest data 
        points on either side). This makes it robust to noise and outliers.
    """
    form1.write(text)
    form1.write('Linear Kernel')

    text = """Formula: K(x_i, x_j) = x_i^T * x_j (essentially the dot product)
        \nWhen to use: Best suited for linearly separable data. Also a 
        good starting point when you're unsure which kernel to choose, 
        as it's generally fast and simple."""
    form1.write(text)
    form1.write('Polynomial Kernel')
    text = """Formula: K(x_i, x_j) = (x_i^T * x_j + 1)^d (where 'd' is the degree of the polynomial)
        \nWhen to use: Useful for problems where the relationship between the data points
        is not simply linear. The degree 'd' controls how flexible the decision boundary can be. """
    form1.write(text)

    form1.write('Radial Basis Function (RBF) Kernel')

    text = """Formula: K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2) (where 'gamma' is 
        a parameter that influences the spread of the kernel)
        \nWhen to use: The RBF kernel is a popular choice as it can 
        handle complex, non-linear decision boundaries. The 'gamma' parameter 
        allows you to control the influence of nearby data points on the decision boundary."""
    form1.write(text)

    submit1 = form1.form_submit_button("Start")

    if submit1:
        form1 = [];
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2
    form2 = st.form("training")
    form2.subheader('Dataset') 
    st.session_state["form2"] = form2    

    df = pd.read_csv('data_decision_trees.csv', header=None)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values   


    form2.write(df)
    form2.write(df.describe().T)


    # insert the rest of the code to train the classifier here        
    form2.write('Display the training result')


    submit2 = form2.form_submit_button("Train")
    if submit2:     

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
        clfSVM = svm.SVC(kernel='rbf', C=1000, gamma=1.0)
        clfSVM.fit(X_train, y_train)
        y_test_pred = clfSVM.predict(X_test)

        form2.subheader('Confusion Matrix')
        form2.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        form2.text(cm)
        form2.subheader('Performance Metrics')
        form2.text(classification_report(y_test, y_test_pred))
        form2.subheader('VIsualization')

        visualize_classifier(clfSVM, X_test, y_test_pred)

        display_form3()

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("prediction")
    form3.subheader('Prediction')
    form3.text('replace with the result of the prediction model.')

    n_clusters = form3.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
    )

    predictbn = form3.form_submit_button("Predict")
    if predictbn:                    
        form3.text('User selected nclusters = ' + str(n_clusters))

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()

def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Specify the title
    ax.set_title(title)
    
    # Choose a color scheme for the plot
    ax.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # Overlay the training points on the plot
    ax.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    # Specify the boundaries of the plot
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    ax.set_xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    ax.set_yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    
    st.session_state["form2"].pyplot(fig)


if __name__ == "__main__":
    app()
