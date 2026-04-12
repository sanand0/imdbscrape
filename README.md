# IMDb Top 250 Scraper: What Broke, Why It Broke, and Why the Schedule Is Off

This repository started as a very small daily scraper for IMDb's Top 250 chart.
It used a plain `httpx` request, parsed the returned HTML with `lxml`, and
appended one JSON line per day to a date-stamped file.

For a while, that worked.

Then it stopped.

This README documents what happened, what changed upstream, what is still true
about the code in this repository, and why the scheduled GitHub Action has been
disabled.

## The Original Design

The scraper in [scrape.py](/home/vscode/code/imdbscrape/scrape.py) is simple:

1. Fetch `https://www.imdb.com/chart/top/`
2. Parse the HTML response
3. Select movie rows with CSS selectors
4. Extract title, year, and rating
5. Append the result to `imdb-top250-YYYY-MM-DD.json`

The script has no browser automation, no cookies, no JavaScript execution, and
no anti-bot handling. That simplicity was a feature when the page was publicly
fetchable as ordinary HTML.

It became the failure mode once the page stopped behaving like ordinary HTML for
this class of client.

## What Failed

Running the script now fails with:

```text
lxml.etree.ParserError: Document is empty
```

That error happens here:

```python
tree = html.fromstring(response.text)
```

At first glance, this looks like a parser bug or a broken selector.
It is neither.

The real failure is earlier: IMDb no longer returns the expected page body to
this request pattern.

## What IMDb Returns Now

As of April 12, 2026, requests from this scraper receive an AWS WAF challenge
response instead of the Top 250 page. In this environment, the key signals are:

- HTTP status: `202`
- Response header: `x-amzn-waf-action: challenge`
- Response body for the script's request path: empty

In some raw `curl` responses, IMDb returns a small interstitial document that
loads AWS WAF challenge JavaScript. In the exact request path used by the
script, the body is empty, which is why `lxml` raises `Document is empty`.

This is the important point: the script is not parsing the wrong HTML. It is no
longer receiving the HTML it expects.

## When It Started

There are two different dates worth keeping separate.

### 1. When AWS introduced WAF challenge support

AWS announced the AWS WAF `Challenge` rule action on October 27, 2022:

- https://aws.amazon.com/about-aws/whats-new/2022/10/aws-waf-challenge-rule-action-bot-control-targeted-bots/

So the underlying mechanism is not new.

### 2. When this repository appears to have been affected

The repository history shows successful daily outputs through:

- `imdb-top250-2026-03-19.json`

And the failure was reproduced in this environment on:

- April 12, 2026

That means the best evidence-based statement is:

IMDb appears to have enabled or tightened AWS WAF challenge behavior for
requests like this sometime after March 19, 2026 and by April 12, 2026.

There is no verified public announcement here giving the exact day IMDb changed
behavior on `/chart/top/`, so anything narrower than that window would be guesswork.

## Why `httpx` No Longer Works

AWS documents the challenge flow as a browser-oriented mechanism.
The challenge page is designed to run JavaScript in the client, obtain a valid
AWS WAF token, and then present that token on subsequent requests.

Relevant AWS documentation:

- AWS WAF JavaScript challenge integration:
  https://docs.aws.amazon.com/waf/latest/developerguide/waf-js-challenge-api.html
- AWS WAF token and domain behavior:
  https://docs.aws.amazon.com/waf/latest/developerguide/web-acl-captcha-challenge-token-domains.html
- AWS WAF `ChallengeAction` API reference:
  https://docs.aws.amazon.com/waf/latest/APIReference/API_ChallengeAction.html

A plain `httpx.get(...)` call does not:

- execute challenge JavaScript
- store and replay the resulting token the way the page expects
- behave like a real browser session

That means the current scraper is not just missing a header or user-agent
string. The model of access is now wrong.

## Why This Is Not Just a Selector Fix

It is tempting to think the page layout changed and the CSS selectors drifted.
That would have been the easy case.

If the page structure had changed, the script would likely still receive HTML,
and one of these things would happen:

- it would return zero movies
- it would extract incomplete fields
- it would fail later while indexing into selectors

Instead, the script fails immediately at HTML parsing because the body is empty.
That points to access control upstream, not DOM drift downstream.

## Why the Daily Workflow Has Been Disabled

The GitHub Action used to run this every day at `00:00 UTC`.

That schedule has been removed for two reasons:

1. The current script is known broken against IMDb's current behavior.
2. Keeping the job on a timer would just generate repeated failing runs with no
   user value.

Manual runs remain enabled through `workflow_dispatch`, which preserves a path
for future testing once the project has a legitimate replacement data source or
a different execution strategy.

## Could We Bypass the Challenge Technically?

Probably, yes.

A real browser automation flow using Playwright or Chromium is more likely to
work than `httpx` because it can:

- load the interstitial
- execute the AWS WAF challenge JavaScript
- obtain the browser token
- continue navigation with normal browser state

But "technically possible" is not the same as "the right fix."

This repository should distinguish between three different questions:

1. Can the challenge be bypassed?
2. Would that be robust?
3. Is that the appropriate or permitted way to obtain the data?

The answer set is not especially flattering:

- It may be possible.
- It will be brittle.
- It may conflict with IMDb's access restrictions and anti-bot posture.

## What IMDb Says About Data Access

IMDb points users toward official data products instead of scraping.

Important references:

- IMDb help, "Can I use IMDb data in my software?":
  https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX
- IMDb Conditions of Use:
  https://www.imdb.com/conditions
- IMDb non-commercial datasets:
  https://developer.imdb.com/non-commercial-datasets/
- IMDb API access documentation:
  https://developer.imdb.com/documentation/api-documentation/getting-access/

The practical reading is straightforward:

- If you need sanctioned non-commercial access, use the published datasets.
- If you need sanctioned live product/API access, use IMDb's official API.
- A brittle HTML scraper against a now-challenged page is not the stable path.

## What the Non-Commercial Datasets Do and Do Not Solve

IMDb publishes non-commercial datasets that include title basics and ratings,
including:

- `title.basics.tsv.gz`
- `title.ratings.tsv.gz`

Those datasets are useful, but they are not a drop-in replacement for the
`/chart/top/` page itself.

Why:

- The Top 250 chart is a curated/ranked IMDb product view.
- The public datasets expose ingredients like rating and vote count.
- They do not simply hand you "the exact current Top 250 page output" as a
  ready-made file.

So a dataset-based rebuild would require a new ranking definition or a best-effort
approximation, and it should be described honestly as that.

## What the Official API Solves

IMDb's official API is the cleanest long-term answer if the goal is current IMDb
data with a stable contract.

That path is stronger because:

- it is sanctioned
- it is designed for structured access
- it avoids HTML scraping fragility
- it aligns with where IMDb is directing developers

The tradeoff is obvious:

- it is a product integration, not a tiny anonymous scrape
- it may require credentials, setup, and potentially paid access

## Current Status of This Repository

Right now, the repository contains:

- the original scraper
- historical daily JSON outputs
- a manual GitHub workflow
- this documentation

It does **not** currently contain a working replacement for the old scraping
path.

That is intentional. A broken scraper should be documented clearly before it is
quietly replaced with something more complicated or more questionable.

## Recommended Next Steps

There are three realistic paths forward.

### Option 1: Rebuild on IMDb's official API

Choose this if the goal is current IMDb data with a stable and legitimate access
method.

This is the best long-term engineering decision.

### Option 2: Rebuild on the non-commercial datasets

Choose this if the project can tolerate a derived or approximate chart based on
published IMDb data rather than the exact `/chart/top/` page.

This is the best free and policy-aligned option.

### Option 3: Use browser automation

Choose this only if the project explicitly accepts the operational fragility and
policy risk of automating a browser through an anti-bot challenge flow.

This is the closest replacement for the old script behavior, but the weakest
long-term design.

## Bottom Line

The script did not fail because of a small bug.
It failed because the assumptions it depended on are no longer true.

This repository used to scrape a publicly fetchable HTML page.
That page is now protected by an AWS WAF challenge flow for this class of
client, and the old one-request parser approach is no longer a valid access
strategy.

That is why the schedule is off, why the scraper is left unmodified, and why any
future fix should start with a decision about data source and access model, not
just parsing code.
