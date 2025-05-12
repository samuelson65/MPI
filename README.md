import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Sample data with a target
data = {
    'diagnosis_codes': [
        "E11.9,I10,J45.909",  # overpaid
        "E11.9,I10",          # overpaid
        "I10,J45.909",        # not overpaid
        "E11.9,I10",          # overpaid
        "J45.909",            # not overpaid
        "E11.9",              # overpaid
        "I10",                # not overpaid
        "E11.9,I10,J45.909",  # overpaid
        "E11.9,J45.909",      # overpaid
        "I10,J45.909"         # not overpaid
    ],
    'overpayment': [1,1,0,1,0,1,0,1,1,0]
}

df = pd.DataFrame(data)

# Step 2: Rule generation function
def get_rules(df_subset, label):
    transactions = df_subset['diagnosis_codes'].apply(lambda x: x.split(',')).tolist()
    te = TransactionEncoder()
    te_data = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)

    # Adjust min_support as needed
    frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    if not rules.empty:
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['rule'] = rules['antecedents_str'] + " -> " + rules['consequents_str']
        rules['group'] = label
    return rules

# Step 3: Get rules for both groups
rules_overpaid = get_rules(df[df['overpayment'] == 1], "Overpaid")
rules_not_overpaid = get_rules(df[df['overpayment'] == 0], "Not Overpaid")

# Step 4: Compare rules
overpaid_rules_set = set(rules_overpaid['rule'])
non_overpaid_rules_set = set(rules_not_overpaid['rule'])

unique_overpaid_rules = overpaid_rules_set - non_overpaid_rules_set
unique_rules_df = rules_overpaid[rules_overpaid['rule'].isin(unique_overpaid_rules)]

# Step 5: Export to Excel
with pd.ExcelWriter("association_rule_comparison.xlsx") as writer:
    if not rules_overpaid.empty:
        rules_overpaid.to_excel(writer, sheet_name="Overpaid Rules", index=False)
    if not rules_not_overpaid.empty:
        rules_not_overpaid.to_excel(writer, sheet_name="Not Overpaid Rules", index=False)
    if not unique_rules_df.empty:
        unique_rules_df.to_excel(writer, sheet_name="Unique Overpaid Rules", index=False)

print("Exported results to 'association_rule_comparison.xlsx'")
