import matplotlib.pyplot as plt

def acc_graph():
    

    # Data for Accuracy 
    accuracy_data = [95.4, 96.5, 95, 97.2, 97.7]

    # X-axis values
    x_values = [0, 500, 1000, 1500, 2000]

    # Creating the bar graph
    plt.bar(x_values, accuracy_data, width=100, align='center', color='green', label='Accuracy')

    # Setting the axes limits
    plt.xlim(0, 2000)
    plt.ylim(0, 100)

    # Adding labels and title
    plt.xlabel('X-Axis - Data Accuracy')
    plt.ylabel('Y-Axis - Percentage')
    plt.title('Proposed')

    # Adding legends
    plt.legend()

    # Displaying the plot
    plt.show()

def prec_graph():
    

    # Data for Precision
    precision_data = [97.7, 95, 97, 96.5, 97.7]

    # X-axis values
    x_values = [0, 500, 1000, 1500, 2000]

    # Creating the bar graph
    plt.bar(x_values, precision_data, width=100, align='edge', color='green', alpha=0.7, label='Precision')
   
    # Setting the axes limits
    plt.xlim(0, 2000)
    plt.ylim(0, 100)

    # Adding labels and title
    plt.xlabel('X-Axis - Data Precision')
    plt.ylabel('Y-Axis - Percentage')
    plt.title('Proposed')

    # Adding legends
    plt.legend()

    # Displaying the plot
    plt.show()

def f1score_graph():    

    # Data for F1 Score
    f1score_data = [96, 96.4, 96.7, 97.2, 97.8]

    # X-axis values
    x_values = [0, 500, 1000, 1500, 2000]

    # Creating the bar graph
    plt.bar(x_values, f1score_data, width=100, align='edge', color='green', alpha=0.7, label='F1 Score')

    # Setting the axes limits
    plt.xlim(0, 2000)
    plt.ylim(0, 100)

    # Adding labels and title
    plt.xlabel('X-Axis - F1 Score')
    plt.ylabel('Y-Axis - Percentage')
    plt.title('Proposed')

    # Adding legends
    plt.legend()

    # Displaying the plot
    plt.show()

def recl_graph():
    

    # Data for Recall
    recall_data = [96, 97.7, 97, 96.7, 97.7]

    # X-axis values
    x_values = [0, 500, 1000, 1500, 2000]

    # Creating the bar graph
    plt.bar(x_values, recall_data, width=100, align='edge', color='green', alpha=0.7, label='Recall')

    # Setting the axes limits
    plt.xlim(0, 2000)
    plt.ylim(0, 100)

    # Adding labels and title
    plt.xlabel('X-Axis - Data Recall')
    plt.ylabel('Y-Axis - Percentage')
    plt.title('Proposed')

    # Adding legends
    plt.legend()

    # Displaying the plot
    plt.show()


def loss_graph():
    

    # Data for Loss
    loss_data = [72.3, 72, 73.4, 74.2, 75]

    # X-axis values
    x_values = [0, 500, 1000, 1500, 2000]

    # Creating the bar graph
    plt.bar(x_values, loss_data, width=100, align='edge', color='green', alpha=0.7, label='Loss')

    # Setting the axes limits
    plt.xlim(0, 2000)
    plt.ylim(0, 100)

    # Adding labels and title
    plt.xlabel('X-Axis - Data Loss')
    plt.ylabel('Y-Axis - Percentage')
    plt.title('Proposed')

    # Adding legends
    plt.legend()

    # Displaying the plot
    plt.show()
