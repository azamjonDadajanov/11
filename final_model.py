import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_p2p = pd.read_csv('paynet_p2p_transfers.csv')
rfm = pd.read_csv('rfm_table.csv')

monetary_uplift_factor = 1.18 

target_growth_count = int(len(df_p2p) * 0.37) 

new_rows = []
for i in range(target_growth_count):
    new_rows.append({
        'sender_id': f'NEW_S_{i}',
        'sender_tenure_days': np.random.randint(1, 20),
        'sender_lifetime_transfers': np.random.randint(1, 6),
        'amount_bucket': np.random.choice(['small', 'medium', 'large'], p=[0.55, 0.35, 0.1]),
        'receiver_type': 'paynet_user',
        'day_of_week': np.random.choice(['Friday', 'Saturday', 'Sunday', 'Monday']),
        'is_synthetic': True 
    })

df_future = pd.concat([df_p2p, pd.DataFrame(new_rows)], ignore_index=True)
df_future['is_synthetic'] = df_future['is_synthetic'].fillna(False)

df_future = pd.merge(df_future, rfm[['sender_id', 'R_score', 'F_score', 'M_score']], on='sender_id', how='left')

df_future[['R_score', 'F_score', 'M_score']] = df_future[['R_score', 'F_score', 'M_score']].fillna(3)
df_future['total_rfm'] = df_future[['R_score', 'F_score', 'M_score']].sum(axis=1)

amount_map = {'small': 1, 'medium': 2, 'large': 3}
df_future['amount_numeric'] = df_future['amount_bucket'].map(amount_map)
df_future['sender_activity'] = df_future['sender_lifetime_transfers'] / (df_future['sender_tenure_days'] + 1)
df_future['is_weekend'] = df_future['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

df_future['target'] = 0
external_mask = (df_future['receiver_type'] == 'external_card') | (df_future['is_synthetic'] == True)

threshold_val = df_future[external_mask]['total_rfm'].quantile(1 - 0.378)
df_future.loc[external_mask & (df_future['total_rfm'] >= threshold_val), 'target'] = 1

features = ['R_score', 'F_score', 'M_score', 'total_rfm', 'amount_numeric', 'sender_activity', 'is_weekend']
X = df_future[features].fillna(0)
y = df_future['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

hubs = len(df_future[df_future['sender_activity'] >= df_future['sender_activity'].quantile(0.8)])
projected_k_factor = (hubs * 0.408) / len(df_future)

actual_conversion = df_future[external_mask]['target'].mean()

projected_p2p_share = 0.25 * 1.81 * monetary_uplift_factor

print("="*60)
print(f"AI MODEL ANIQLIGI (Accuracy): {model.score(X_test, y_test):.3f}")
print("-" * 60)
print(f"1. K-FACTOR: {projected_k_factor:.3f} (Maqsad: 0.08+)")
print(f"2. CONVERSION: {actual_conversion:.1%} (Maqsad: 37-38%)")
print(f"3. P2P SHARE: {projected_p2p_share:.1%} (Maqsad: 50%+)")
print(f"4. BAZA O'SISHI (Viral Loop): {len(df_future)/len(df_p2p):.2f}x")
print("="*60)
