#!/usr/bin/env python3
"""
Deep investigative analysis of IMDb Top 250 data.
Hunting for surprising insights that challenge conventional wisdom.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

def parse_rating(rating_str):
    """Parse rating string like '9.3\xa0(3.1M)' into (score, votes)"""
    rating_str = rating_str.replace('\xa0', ' ')
    match = re.match(r'([\d.]+)\s*\(([\d.]+)([KM])\)', rating_str)
    if match:
        score = float(match.group(1))
        vote_num = float(match.group(2))
        multiplier = match.group(3)
        if multiplier == 'M':
            votes = int(vote_num * 1_000_000)
        elif multiplier == 'K':
            votes = int(vote_num * 1_000)
        return score, votes
    return None, None

def parse_title(title_str):
    """Parse title like '1. The Shawshank Redemption' into (rank, name)"""
    match = re.match(r'(\d+)\.\s*(.+)', title_str)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, title_str

def normalize_movie_name(name):
    """Normalize movie names to handle variations"""
    # Common variations
    name = name.replace('GoodFellas', 'Goodfellas')
    name = name.replace('The Godfather Part II', 'The Godfather: Part II')
    return name

def load_all_data():
    """Load all daily snapshots"""
    data_dir = Path('/home/user/imdbscrape')
    all_data = []

    for json_file in sorted(data_dir.glob('imdb-top250-*.json')):
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', json_file.name)
        if date_match:
            date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()

            with open(json_file) as f:
                # Take only the first line (first snapshot of the day)
                first_line = f.readline().strip()
                if first_line:
                    snapshot = json.loads(first_line)
                    all_data.append((date, snapshot))

    return sorted(all_data, key=lambda x: x[0])

def analyze_vote_growth():
    """Analyze vote growth patterns - looking for anomalies"""
    data = load_all_data()

    # Track votes over time for each movie
    movie_votes = defaultdict(list)
    movie_ratings = defaultdict(list)

    for date, snapshot in data:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            score, votes = parse_rating(movie['rating'])

            if votes:
                movie_votes[name].append((date, votes))
                movie_ratings[name].append((date, score))

    # Calculate daily vote growth rates
    growth_rates = {}
    for movie, vote_series in movie_votes.items():
        if len(vote_series) > 30:  # Need enough data points
            daily_changes = []
            for i in range(1, len(vote_series)):
                days = (vote_series[i][0] - vote_series[i-1][0]).days
                if days > 0:
                    daily_change = (vote_series[i][1] - vote_series[i-1][1]) / days
                    daily_changes.append(daily_change)

            if daily_changes:
                growth_rates[movie] = {
                    'mean': statistics.mean(daily_changes),
                    'median': statistics.median(daily_changes),
                    'std': statistics.stdev(daily_changes) if len(daily_changes) > 1 else 0,
                    'total_votes_start': vote_series[0][1],
                    'total_votes_end': vote_series[-1][1],
                    'total_growth': vote_series[-1][1] - vote_series[0][1],
                    'days': (vote_series[-1][0] - vote_series[0][0]).days,
                    'data_points': len(vote_series),
                }

    return growth_rates, movie_votes, movie_ratings

def analyze_rank_stability():
    """Which movies have the most volatile rankings?"""
    data = load_all_data()

    movie_ranks = defaultdict(list)

    for date, snapshot in data:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            movie_ranks[name].append((date, rank))

    # Calculate rank volatility
    volatility = {}
    for movie, rank_series in movie_ranks.items():
        if len(rank_series) > 30:
            ranks = [r[1] for r in rank_series]
            volatility[movie] = {
                'mean_rank': statistics.mean(ranks),
                'min_rank': min(ranks),
                'max_rank': max(ranks),
                'rank_range': max(ranks) - min(ranks),
                'std': statistics.stdev(ranks) if len(ranks) > 1 else 0,
                'appearances': len(rank_series),
                'first_date': rank_series[0][0],
                'last_date': rank_series[-1][0],
            }

    return volatility, movie_ranks

def analyze_rating_changes():
    """Track actual rating score changes - these are rare and meaningful"""
    data = load_all_data()

    movie_ratings = defaultdict(list)
    rating_changes = []

    for date, snapshot in data:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            score, votes = parse_rating(movie['rating'])

            if score:
                movie_ratings[name].append((date, score, votes))

    # Find rating changes (these are rare!)
    for movie, series in movie_ratings.items():
        for i in range(1, len(series)):
            if series[i][1] != series[i-1][1]:
                rating_changes.append({
                    'movie': movie,
                    'date': series[i][0],
                    'old_rating': series[i-1][1],
                    'new_rating': series[i][1],
                    'votes_at_change': series[i][2],
                    'change': series[i][1] - series[i-1][1],
                })

    return rating_changes, movie_ratings

def analyze_newcomers_and_exits():
    """Track which movies entered and left the top 25"""
    data = load_all_data()

    daily_top25 = {}
    for date, snapshot in data:
        movies_that_day = set()
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            movies_that_day.add(name)
        daily_top25[date] = movies_that_day

    # Find entries and exits
    dates = sorted(daily_top25.keys())
    entries = []
    exits = []

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]

        new_movies = daily_top25[curr_date] - daily_top25[prev_date]
        lost_movies = daily_top25[prev_date] - daily_top25[curr_date]

        for movie in new_movies:
            entries.append((curr_date, movie))
        for movie in lost_movies:
            exits.append((curr_date, movie))

    return entries, exits, daily_top25

def analyze_vote_efficiency():
    """Calculate 'vote efficiency' - votes per rank position"""
    data = load_all_data()

    # Get latest snapshot
    latest_date, latest_snapshot = data[-1]

    efficiency = []
    for movie in latest_snapshot['movies']:
        rank, name = parse_title(movie['title'])
        name = normalize_movie_name(name)
        score, votes = parse_rating(movie['rating'])

        if votes and rank:
            efficiency.append({
                'movie': name,
                'rank': rank,
                'votes': votes,
                'rating': score,
                'votes_per_rank': votes / rank,
            })

    return efficiency

def analyze_decade_representation():
    """What decades dominate the top 25?"""
    data = load_all_data()

    decade_counts = defaultdict(lambda: defaultdict(int))

    for date, snapshot in data:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            year = int(movie['year'])
            decade = (year // 10) * 10
            decade_counts[date][decade] += 1

    return decade_counts

def find_suspicious_patterns():
    """Look for statistically suspicious patterns"""
    growth_rates, movie_votes, movie_ratings = analyze_vote_growth()

    # Look for movies with unusually high or low growth
    if growth_rates:
        mean_growth = statistics.mean([g['mean'] for g in growth_rates.values()])
        std_growth = statistics.stdev([g['mean'] for g in growth_rates.values()])

        outliers = []
        for movie, stats in growth_rates.items():
            z_score = (stats['mean'] - mean_growth) / std_growth if std_growth > 0 else 0
            if abs(z_score) > 2:
                outliers.append({
                    'movie': movie,
                    'z_score': z_score,
                    'daily_growth': stats['mean'],
                    'total_growth': stats['total_growth'],
                })

        return outliers, mean_growth, std_growth
    return [], 0, 0

def main():
    print("=" * 80)
    print("IMDb TOP 250 INVESTIGATIVE ANALYSIS")
    print("=" * 80)

    # 1. VOTE GROWTH ANALYSIS
    print("\n### 1. VOTE GROWTH PATTERNS ###\n")
    growth_rates, movie_votes, _ = analyze_vote_growth()

    # Sort by daily vote growth
    sorted_growth = sorted(growth_rates.items(), key=lambda x: x[1]['mean'], reverse=True)

    print("TOP 10 - FASTEST VOTE GROWTH (votes/day):")
    for movie, stats in sorted_growth[:10]:
        print(f"  {movie}")
        print(f"    Daily: +{stats['mean']:,.0f} votes/day")
        print(f"    Total: +{stats['total_growth']:,} votes over {stats['days']} days")
        print(f"    From {stats['total_votes_start']:,} to {stats['total_votes_end']:,}")
        growth_pct = (stats['total_growth'] / stats['total_votes_start']) * 100
        print(f"    Growth %: {growth_pct:.2f}%")
        print()

    # 2. RATING CHANGES - THESE ARE THE BIG STORY
    print("\n### 2. RATING SCORE CHANGES ###\n")
    rating_changes, movie_ratings = analyze_rating_changes()

    if rating_changes:
        print(f"Total rating changes observed: {len(rating_changes)}")
        print("\nAll rating changes (chronological):")
        for change in sorted(rating_changes, key=lambda x: x['date']):
            direction = "⬆️" if change['change'] > 0 else "⬇️"
            print(f"  {change['date']}: {change['movie']}")
            print(f"    {change['old_rating']} → {change['new_rating']} ({direction} {change['change']:+.1f})")
            print(f"    Votes at change: {change['votes_at_change']:,}")
            print()
    else:
        print("No rating changes detected!")

    # 3. RANK VOLATILITY
    print("\n### 3. RANK VOLATILITY ###\n")
    volatility, movie_ranks = analyze_rank_stability()

    # Most volatile
    sorted_volatile = sorted(volatility.items(), key=lambda x: x[1]['std'], reverse=True)
    print("MOST VOLATILE RANKINGS (highest std dev):")
    for movie, stats in sorted_volatile[:10]:
        print(f"  {movie}")
        print(f"    Range: #{stats['min_rank']} to #{stats['max_rank']} (span of {stats['rank_range']})")
        print(f"    Std Dev: {stats['std']:.2f}")
        print(f"    Mean Rank: {stats['mean_rank']:.1f}")
        print()

    # Most stable
    print("MOST STABLE RANKINGS (lowest std dev):")
    sorted_stable = sorted(volatility.items(), key=lambda x: x[1]['std'])
    for movie, stats in sorted_stable[:5]:
        print(f"  {movie}")
        print(f"    Always at rank #{stats['min_rank']}-{stats['max_rank']}")
        print(f"    Std Dev: {stats['std']:.4f}")
        print()

    # 4. NEWCOMERS AND EXITS
    print("\n### 4. TOP 25 ENTRIES AND EXITS ###\n")
    entries, exits, daily_top25 = analyze_newcomers_and_exits()

    print(f"Movies that ENTERED top 25: {len(entries)}")
    for date, movie in entries:
        print(f"  {date}: {movie}")

    print(f"\nMovies that LEFT top 25: {len(exits)}")
    for date, movie in exits:
        print(f"  {date}: {movie}")

    # 5. VOTE EFFICIENCY
    print("\n### 5. VOTE EFFICIENCY (current) ###\n")
    efficiency = analyze_vote_efficiency()

    # Sort by votes to see if there's a correlation with rank
    print("Votes vs Rank (current snapshot):")
    for item in efficiency[:25]:
        print(f"  #{item['rank']:2d} {item['movie'][:40]:40s} {item['votes']:>10,} votes  {item['rating']}")

    # 6. DECADE ANALYSIS
    print("\n### 6. DECADE REPRESENTATION ###\n")
    decade_counts = analyze_decade_representation()

    # Get first and last snapshots
    dates = sorted(decade_counts.keys())
    first_date = dates[0]
    last_date = dates[-1]

    print(f"First snapshot ({first_date}):")
    for decade in sorted(decade_counts[first_date].keys()):
        print(f"  {decade}s: {decade_counts[first_date][decade]} movies")

    print(f"\nLatest snapshot ({last_date}):")
    for decade in sorted(decade_counts[last_date].keys()):
        print(f"  {decade}s: {decade_counts[last_date][decade]} movies")

    # 7. SUSPICIOUS PATTERNS
    print("\n### 7. STATISTICAL OUTLIERS ###\n")
    outliers, mean_growth, std_growth = find_suspicious_patterns()

    print(f"Mean daily vote growth: {mean_growth:,.0f} votes/day")
    print(f"Std deviation: {std_growth:,.0f} votes/day")
    print(f"\nOutliers (|z-score| > 2):")
    for outlier in sorted(outliers, key=lambda x: abs(x['z_score']), reverse=True):
        print(f"  {outlier['movie']}")
        print(f"    Z-score: {outlier['z_score']:+.2f}")
        print(f"    Daily growth: {outlier['daily_growth']:,.0f} votes/day")
        print()

    # 8. THE BIG STORY: Calculate the "Impossible" movies
    print("\n### 8. THE PARADOX: HIGH VOTES BUT LOW RANK ###\n")
    efficiency = analyze_vote_efficiency()

    # Find movies that break the expected pattern
    votes_sorted = sorted(efficiency, key=lambda x: x['votes'], reverse=True)

    print("Movies sorted by votes (descending):")
    for i, item in enumerate(votes_sorted[:25], 1):
        mismatch = item['rank'] - i
        if mismatch > 3:
            flag = f" ⚠️ EXPECTED RANK #{i}, ACTUAL #{item['rank']} (off by {mismatch})"
        else:
            flag = ""
        print(f"  {i:2d}. {item['movie'][:35]:35s} {item['votes']:>10,} votes  (actual #{item['rank']}){flag}")

if __name__ == "__main__":
    main()
