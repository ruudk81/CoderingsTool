python cleanup/cleanup.py              # full run with all 4 stages
python cleanup/cleanup.py --dry-run    # preview only
python cleanup/cleanup.py --skip-git   # skip git stage, run stages 2-4
python cleanup/cleanup.py --age 14     # use 14-day threshold instead of 30