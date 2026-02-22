python devs/cleanup.py              # full run with all 4 stages
python devs/cleanup.py --dry-run    # preview only
python devs/cleanup.py --skip-git   # skip git stage, run stages 2-4
python devs/cleanup.py --age 14     # use 14-day threshold instead of 30