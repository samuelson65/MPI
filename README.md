import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Sample data with target variable
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

# Step 2: Function to get rules for a group
def get_rules(df_subset, label):
    transactions = df_subset['diagnosis_codes'].apply(lambda x: x.split(',')).tolist()
    te = TransactionEncoder()
    te_data = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(x))
        rules['group'] = label
    return rules

# Step 3: Generate rules for both groups
rules_overpaid = get_rules(df[df['overpayment'] == 1], "Overpaid")
rules_nonoverpaid = get_rules(df[df['overpayment'] == 0], "Not Overpaid")

# Step 4: Combine and compare
all_rules = pd.concat([rules_overpaid, rules_nonoverpaid], ignore_index=True)

# Optional: Identify rules unique to overpaid
unique_overpaid = set(rules_overpaid['antecedents'] + "->" + rules_overpaid['consequents'])
unique_nonoverpaid = set(rules_nonoverpaid['antecedents'] + "->" + rules_nonoverpaid['consequents'])

only_in_overpaid = unique_overpaid - unique_nonoverpaid

print("\nUnique Patterns in Overpaid Claims:")
for rule in only_in_overpaid:
    print(rule)

# Step 5: Export to Excel
with pd.ExcelWriter("comparison_association_rules.xlsx") as writer:
    rules_overpaid.to_excel(writer, sheet_name="Overpaid Rules", index=False)
    rules_nonoverpaid.to_excel(writer, sheet_name="Non-Overpaid Rules", index=False)
    pd.DataFrame({'Overpaid_Only_Rules': list(only_in_overpaid)}).to_excel(writer, sheet_name="Unique Overpaid", index=False)

print("\nExported to 'comparison_association_rules.xlsx'")
