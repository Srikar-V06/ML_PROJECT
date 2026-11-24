# Decision memo | Head of Risk

**Subject — Loan approval thresholds under three loss assumptions**

At a 10:1 loss ratio, the validation-selected logreg_baseline policy approved 30.0% of the 30,000 later test loans. Its cost was 697.0 units per 1,000 applicants, a reduction of 1342.7 against approving everyone. Each unit is the assumed cost of rejecting one borrower who would have repaid.

| Default approval cost / good rejection cost | Reject at or above | Test approval | Cost per 1,000 | At 0.50 | Approve all |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5:1 | 0.14321 | 47.9% | 572.7 | 952.4 | 1,019.8 |
| 10:1 | 0.09684 | 30.0% | 697.0 | 1,894.4 | 2,039.7 |
| 15:1 | 0.07077 | 19.8% | 740.6 | 2,836.4 | 3,059.5 |

All three policies used thresholds selected before Test. The assumed loss on a default raises the cost of approving risky loans and lowers the approval rate. The observed Test default rate was 20.40%.

To translate these results into money, multiply the table by the institution's estimated loss from rejecting a good loan. The analysis assigns the same exposure to each application. Vintage censoring and the absence of rejected applicants limit these estimates to the historical accepted-loan cohort. A rollout decision needs loss amounts, margins, and a prospective check of approval volume and calibration.
