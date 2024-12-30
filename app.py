import streamlit as st
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit 


##python -m streamlit run C:\Users\angel\Documents\GitHub\Data-Visualization-Website\app.py --theme.primaryColor="#ffe3e1" --theme.backgroundColor="#fff5e4" --theme.secondaryBackgroundColor="#ecdfeb" --theme.textColor="#ff9494" --theme.font="monospace"     

with st.sidebar: 
    option = st.selectbox(
    "Select how you wish to upload data",
    ("Manual Input", "CSV file"),
    index=None,
    placeholder="Select option...",
    )


if option == "Manual Input":
    
    with st.sidebar:
        x_val = st.text_area(
            "Enter your X values",
            placeholder="Example: 1, 2, 3, 4, 5",
        )

        y_val = st.text_area(
            "Enter your Y values",
            placeholder="Example: 6, 7, 8, 9",
        )

        with st.sidebar:
            best_fit = st.selectbox(
            "Select line of best of fit",
            ("Polynomial", "Degree", "Exponential", "logarithmic"), #Erm learn how to spell 
            index=None,
            placeholder="Select option...",
        )

    try:
        if x_val and y_val:
            x_list = [float(val.strip()) for val in x_val.split(",") if val.strip()]
            y_list = [float(val.strip()) for val in y_val.split(",") if val.strip()]

            x = np.array(x_list)
            y = np.array(y_list)

            if len(x) != len(y):
                st.error("X and Y values must have the same number of elements.")
                st.markdown("![Alt Text](https://media1.tenor.com/m/b9M-lttDYyYAAAAC/uhh-cat.gif)")

            else:
                st.header("Here's your graph")

                fig, ax = plt.subplots()
                ax.scatter(x, y, color="pink")
                ax.set_xlabel("X-axis")
                ax.set_ylabel("Y-axis") 

                avg_error = None 
                max_error = None

                if best_fit == "Polynomial":  
                    with st.sidebar: 
                        polynomial = st.selectbox(
                        "Select the degree of polynomial",
                        ("Linear", "Quadratic", "Cubic", "Quartic", "Quintic"), #Erm learn how to spell 
                        index=None, placeholder="Select option...",
                        )
                        if polynomial == "Linear":
                            a, b = np.polyfit(x, y, deg=1)
                            plt.plot(x, a*x + b, label=f'y = {a:.2f}x + {b:.2f}', linestyle='--')
                            plt.legend()

                            y_fit = a*x + b
                            residuals = y - y_fit

                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                        elif polynomial == "Quadratic":
                            a, b, c = np.polyfit(x, y, deg=2)
                            x_curve = np.linspace(min(x), max(x), 500)
                            y_curve = a * x_curve**2 + b * x_curve + c
                            plt.scatter(x, y, color="pink")
                            plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x² + {b:.2f}x + {c:.2f}', linestyle='--')
                            plt.legend()

                            y_fit = a * x**2 + b * x + c
                            residuals = y - y_fit
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                        elif polynomial == "Cubic":
                            a, b, c, d = np.polyfit(x, y, deg=3)
                            x_curve = np.linspace(min(x), max(x), 500)
                            y_curve = a * x_curve**3 + b * x_curve**2 + c * x_curve + d
                            plt.scatter(x, y, color="pink")
                            plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x³ + {b:.2f}x² + {c:.2f}x + {d:.2f}', linestyle='--')
                            plt.legend()

                            y_fit = a * x**3 + b * x**2 + c * x + d
                            residuals = y - y_fit
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                        elif polynomial == "Quartic":
                            a, b, c, d, e = np.polyfit(x, y, deg=4)
                            x_curve = np.linspace(min(x), max(x), 500)
                            y_curve = a * x_curve**4 + b * x_curve**3 + c * x_curve**2 + d * x_curve + e
                            plt.scatter(x, y, color="pink")
                            plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x⁴ + {b:.2f}x³ + {c:.2f}x² + {d:.2f}x + {e:.2f}', linestyle='--')
                            plt.legend()

                            y_fit = a * x**4 + b * x**3 + c * x**2 + d * x + e
                            residuals = y - y_fit
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                        elif polynomial == "Quintic":
                            a, b, c, d, e, f = np.polyfit(x, y, deg=5)
                            x_curve = np.linspace(min(x), max(x), 500)
                            y_curve = a * x_curve**5 + b * x_curve**4 + c * x_curve**3 + d * x_curve**2 + e * x_curve + f
                            plt.scatter(x, y, color="pink")
                            plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x⁵ + {b:.2f}x⁴ + {c:.2f}x³ + {d:.2f}x² + {e:.2f}x + {f:.2f}', linestyle='--')
                            plt.legend()

                            y_fit = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
                            residuals = y - y_fit
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                elif best_fit == "Degree": 
                    with st.sidebar: 
                        degree = st.slider("Please select your degree", 1, 5, 1)
                        coefficients = np.polyfit(x, y, deg=degree)

                        # Generate fitted values for the actual x points
                        y_fit_at_x = np.polyval(coefficients, x)

                        # Generate a smooth curve for visualization
                        x_fit = np.linspace(min(x), max(x), 500)
                        y_fit = np.polyval(coefficients, x_fit)

                        # Calculate residuals and error metrics
                        residuals = y - y_fit_at_x
                        avg_error = np.mean(np.abs(residuals))
                        max_error = np.max(np.abs(residuals))

                        # Generate the polynomial equation string
                        equation_terms = [
                            f"{coeff:.2f}x^{degree - i}" if degree - i > 0 else f"{coeff:.2f}"
                            for i, coeff in enumerate(coefficients)
                        ]
                        equation = " + ".join(equation_terms)

                        # Plot the data and the fitted curve
                        plt.scatter(x, y, color="pink")
                        plt.plot(x_fit, y_fit, linestyle="--", label=f"y = {equation}")
                        plt.legend()

                elif best_fit == "Exponential":
                    
                    def exponential(x, a, b):
                        return a * np.exp(b * x)
                
                    params, covariance = curve_fit(exponential, x, y)
                    a, b = params

                    y_fit_at_x = exponential(x, a, b)

                    x_fit = np.linspace(min(x), max(x), 500)
                    y_fit = exponential(x_fit, a, b)
                    
                    plt.scatter(x, y, color="pink")
                    plt.plot(x_fit, y_fit, label=f"y = {a:.2f}e^({b:.2f}x)", linestyle='--')
                    plt.legend()

                    residuals = y - y_fit_at_x
                    avg_error = np.mean(np.abs(residuals))
                    max_error = np.max(np.abs(residuals))

                elif best_fit == "logarithmic":
                    def logarithmic(x, a, b):
                        return a * np.log(x) + b

                    if np.any(x <= 0):
                        st.error("Logarithmic fit is only valid for positive X values. Please ensure all X values are greater than zero.")
                    else:
                        params, covariance = curve_fit(logarithmic, x, y)
                        a, b = params

                        y_fit_at_x = logarithmic(x, a, b)

                        x_fit = np.linspace(min(x), max(x), 500)
                        y_fit = logarithmic(x_fit, a, b)

                        residuals = y - y_fit_at_x
                        avg_error = np.mean(np.abs(residuals))
                        max_error = np.max(np.abs(residuals))

                        plt.scatter(x, y, color="pink")
                        plt.plot(x_fit, y_fit, label=f"y = {a:.2f}ln(x) + {b:.2f}", linestyle='--')
                        plt.legend()
            
                st.pyplot(fig)  

                if avg_error is not None and max_error is not None:
                    st.write(f"Average Error: {avg_error:.4f}")
                    st.write(f"Maximum Error: {max_error:.4f}")
                

        else:
            st.info("Please enter both X and Y values to plot the graph.")
            
            

    except ValueError as e:
        st.error(f"Invalid input: {e}. Please ensure all values are numeric and separated by commas.")
        st.markdown("![Alt Text](https://media1.tenor.com/m/R514o0vP7kEAAAAd/weird-stare.gif)")

if option == "CSV file":
    
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Choose a CSV file", accept_multiple_files=True
            )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)

                st.write(f"Data from {uploaded_file.name}:")
                st.write(df)

                columns = df.columns.tolist()
                x_col = st.selectbox(f"Select X-axis for {uploaded_file.name}:", columns)
                y_col = st.selectbox(f"Select Y-axis for {uploaded_file.name}:", columns)
                st.write("Try to select the bigger number set as the y-axis or else Exponential will not work")

                fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis
                ax.scatter(df[x_col], df[y_col], color="pink")       # Plot the scatter plot on the axis
                ax.set_title(f"Scatter Plot of {y_col} vs {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True)

                with st.sidebar:
                    best_fit = st.selectbox(
                    "Select line of best of fit",
                    ("Polynomial", "Degree", "Exponential", "logarithmic"), #Erm learn how to spell 
                    index=None,
                    placeholder="Select option...",
                    )
                    y = df[y_col]
                    x = df[x_col]

                    avg_error = None
                    max_error = None

                    if best_fit == "Polynomial":  
                        with st.sidebar: 
                            polynomial = st.selectbox(
                            "Select the degree of polynomial",
                            ("Linear", "Quadratic", "Cubic", "Quartic", "Quintic"), #Erm learn how to spell 
                            index=None, placeholder="Select option...",
                            )
                            if polynomial == "Linear":
                                a, b = np.polyfit(x, y, deg=1)
                                plt.scatter(x, y, color="pink")
                                plt.plot(x, a*x + b, label=f'y = {a:.2f}x + {b:.2f}', linestyle='--')
                                plt.legend()

                                y_fit = a*x + b
                                residuals = y - y_fit

                                avg_error = np.mean(np.abs(residuals))
                                max_error = np.max(np.abs(residuals))

                            elif polynomial == "Quadratic":
                                a, b, c = np.polyfit(x, y, deg=2)
                                x_curve = np.linspace(min(x), max(x), 500)
                                y_curve = a * x_curve**2 + b * x_curve + c
                                plt.scatter(x, y, color="pink")
                                plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x² + {b:.2f}x + {c:.2f}', linestyle='--')
                                plt.legend()

                                y_fit = a * x**2 + b * x + c
                                residuals = y - y_fit
                                avg_error = np.mean(np.abs(residuals))
                                max_error = np.max(np.abs(residuals))

                            elif polynomial == "Cubic":
                                a, b, c, d = np.polyfit(x, y, deg=3)
                                x_curve = np.linspace(min(x), max(x), 500)
                                y_curve = a * x_curve**3 + b * x_curve**2 + c * x_curve + d
                                plt.scatter(x, y, color="pink")
                                plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x³ + {b:.2f}x² + {c:.2f}x + {d:.2f}', linestyle='--')
                                plt.legend()

                                y_fit = a * x**3 + b * x**2 + c * x + d
                                residuals = y - y_fit
                                avg_error = np.mean(np.abs(residuals))
                                max_error = np.max(np.abs(residuals))

                            elif polynomial == "Quartic":
                                a, b, c, d, e = np.polyfit(x, y, deg=4)
                                x_curve = np.linspace(min(x), max(x), 500)
                                y_curve = a * x_curve**4 + b * x_curve**3 + c * x_curve**2 + d * x_curve + e
                                plt.scatter(x, y, color="pink")
                                plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x⁴ + {b:.2f}x³ + {c:.2f}x² + {d:.2f}x + {e:.2f}', linestyle='--')
                                plt.legend()

                                y_fit = a * x**4 + b * x**3 + c * x**2 + d * x + e
                                residuals = y - y_fit
                                avg_error = np.mean(np.abs(residuals))
                                max_error = np.max(np.abs(residuals))

                            elif polynomial == "Quintic":
                                a, b, c, d, e, f = np.polyfit(x, y, deg=5)
                                x_curve = np.linspace(min(x), max(x), 500)
                                y_curve = a * x_curve**5 + b * x_curve**4 + c * x_curve**3 + d * x_curve**2 + e * x_curve + f
                                plt.scatter(x, y, color="pink")
                                plt.plot(x_curve, y_curve, label=f'y = {a:.2f}x⁵ + {b:.2f}x⁴ + {c:.2f}x³ + {d:.2f}x² + {e:.2f}x + {f:.2f}', linestyle='--')
                                plt.legend()

                                y_fit = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
                                residuals = y - y_fit
                                avg_error = np.mean(np.abs(residuals))
                                max_error = np.max(np.abs(residuals))

                    elif best_fit == "Degree": 
                        with st.sidebar: 
                            degree = st.slider("Please select your degree", 1, 5, 1)
                            coefficients = np.polyfit(x, y, deg=degree)

                            # Generate fitted values for the actual x points
                            y_fit_at_x = np.polyval(coefficients, x)

                            # Generate a smooth curve for visualization
                            x_fit = np.linspace(min(x), max(x), 500)
                            y_fit = np.polyval(coefficients, x_fit)

                            # Calculate residuals and error metrics
                            residuals = y - y_fit_at_x
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                            # Generate the polynomial equation string
                            equation_terms = [
                                f"{coeff:.2f}x^{degree - i}" if degree - i > 0 else f"{coeff:.2f}"
                                for i, coeff in enumerate(coefficients)
                            ]
                            equation = " + ".join(equation_terms)

                            # Plot the data and the fitted curve
                            plt.scatter(x, y, color="pink")
                            plt.plot(x_fit, y_fit, linestyle="--", label=f"y = {equation}")
                            plt.legend()

                    elif best_fit == "Exponential":

                        x = np.array(x)
                        y = np.array(y)
                        
                        def exponential(x, a, b):
                            return a * np.exp(b * x)
                    
                        params, covariance = curve_fit(exponential, x, y)
                        a, b = params

                        y_fit_at_x = exponential(x, a, b)

                        x_fit = np.linspace(min(x), max(x), 500)
                        y_fit = exponential(x_fit, a, b)

                        plt.scatter(x, y, color="pink")
                        plt.plot(x_fit, y_fit, label=f"y = {a:.2f}e^({b:.2f}x)", linestyle='--')
                        plt.legend()

                        residuals = y - y_fit_at_x
                        avg_error = np.mean(np.abs(residuals))
                        max_error = np.max(np.abs(residuals))

                    elif best_fit == "logarithmic":
                        def logarithmic(x, a, b):
                            return a * np.log(x) + b

                        if np.any(x <= 0):
                            st.error("Logarithmic fit is only valid for positive X values. Please ensure all X values are greater than zero.")
                        else:
                            params, covariance = curve_fit(logarithmic, x, y)
                            a, b = params

                            y_fit_at_x = logarithmic(x, a, b)

                            x_fit = np.linspace(min(x), max(x), 500)
                            y_fit = logarithmic(x_fit, a, b)

                            residuals = y - y_fit_at_x
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))

                            plt.scatter(x, y, color="pink")
                            plt.plot(x_fit, y_fit, label=f"y = {a:.2f}ln(x) + {b:.2f}", linestyle='--')
                            plt.legend()

                    
                
                st.pyplot(fig)
                if avg_error is not None and max_error is not None:
                        st.write(f"Average Error: {avg_error:.4f}")
                        st.write(f"Maximum Error: {max_error:.4f}")

            except ValueError as e:
                st.error(f"Invalid input in {uploaded_file.name}: {e}. Please upload a valid CSV file.")
    else:
        st.warning("Please upload at least one CSV file.")

if option == None:
    st.title("Curve Fitting App")
    st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")
    st.text("Made by Angela with Love 🌷")

