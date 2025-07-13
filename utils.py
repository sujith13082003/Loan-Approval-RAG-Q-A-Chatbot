import pandas as pd

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(subset=["Loan_ID"], inplace=True)  # remove duplicate rows
    df.fillna("Unknown", inplace=True)

    def row_to_text(row):
        return (
            f"Applicant {row['Loan_ID']} is a {row['Gender']} {row['Married']} "
            f"{row['Education']} who is {row['Self_Employed']} and has a credit history of {row['Credit_History']}. "
            f"The loan amount requested is {row['LoanAmount']} with a term of {row['Loan_Amount_Term']}. "
            f"Loan status: {row['Loan_Status']}."
        )

    return df.apply(row_to_text, axis=1).tolist()