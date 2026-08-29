# Annotation guide

Assign every label directly supported by the screenshot. Labels are multi-select.

- **credential_theft**: authentication secrets: user/email + password, PIN, OTP, recovery answer/code.
- **financial_fraud**: card/bank/payment/transfer/investment or direct monetary solicitation.
- **malware_distribution**: prompts to download or install a file, app, update or executable.
- **personal_information_harvesting**: phone, physical address, date of birth, government ID, employment or other PII beyond login.

Do not label on brand alone. If image quality prevents a decision, mark `exclude`. If no intention is visible, mark `labelled` with all labels false and explain in notes. Recommended process: annotator A and B label independently; adjudicator resolves differences and writes the final manifest. Track Cohen's kappa per label in your report.
