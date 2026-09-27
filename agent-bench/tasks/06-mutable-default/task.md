Our request handlers leak state between requests: headers added for one request show up on later, unrelated requests. Track down why and fix it. Run `python -m pytest -q`.
