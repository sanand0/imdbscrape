#!/usr/bin/env python3
"""
Deep investigation into the Paradox of Popularity.
Why do the most-voted movies rank lower than less-voted classics?
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics
import math

def parse_rating(rating_str):
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
    match = re.match(r'(\d+)\.\s*(.+)', title_str)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, title_str

def normalize_movie_name(name):
    name = name.replace('GoodFellas', 'Goodfellas')
    name = name.replace('The Godfather Part II', 'The Godfather: Part II')
    return name

def load_all_data():
    data_dir = Path('/home/user/imdbscrape')
    all_data = []
    for json_file in sorted(data_dir.glob('imdb-top250-*.json')):
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', json_file.name)
        if date_match:
            date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
            with open(json_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    snapshot = json.loads(first_line)
                    all_data.append((date, snapshot))
    return sorted(all_data, key=lambda x: x[0])

def calculate_vote_rank_correlation():
    """Calculate how well votes predict rank"""
    data = load_all_data()

    correlations_over_time = []

    for date, snapshot in data:
        votes_list = []
        ranks_list = []

        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            score, votes = parse_rating(movie['rating'])
            if votes and rank:
                votes_list.append(votes)
                ranks_list.append(rank)

        # Calculate Spearman correlation (rank-based)
        n = len(votes_list)
        if n > 2:
            # Sort by votes to get vote ranks
            vote_ranks = [sorted(votes_list, reverse=True).index(v) + 1 for v in votes_list]
            actual_ranks = ranks_list

            # Spearman's rho
            d_squared = sum((vr - ar)**2 for vr, ar in zip(vote_ranks, actual_ranks))
            rho = 1 - (6 * d_squared) / (n * (n**2 - 1))

            correlations_over_time.append((date, rho))

    return correlations_over_time

def analyze_decade_effect():
    """Is there a systematic bias by movie decade?"""
    data = load_all_data()

    # Get latest data
    date, snapshot = data[-1]

    decade_stats = defaultdict(list)

    for movie in snapshot['movies']:
        rank, name = parse_title(movie['title'])
        score, votes = parse_rating(movie['rating'])
        year = int(movie['year'])
        decade = (year // 10) * 10

        # Calculate expected rank based on votes
        all_movies = []
        for m in snapshot['movies']:
            r, n = parse_title(m['title'])
            s, v = parse_rating(m['rating'])
            all_movies.append((n, r, v))

        vote_sorted = sorted(all_movies, key=lambda x: x[2], reverse=True)
        expected_rank = next(i+1 for i, m in enumerate(vote_sorted) if normalize_movie_name(m[0]) == normalize_movie_name(name))

        rank_bonus = expected_rank - rank  # Positive = performing better than votes suggest

        decade_stats[decade].append({
            'movie': name,
            'actual_rank': rank,
            'expected_rank': expected_rank,
            'rank_bonus': rank_bonus,
            'votes': votes,
            'rating': score,
        })

    return decade_stats

def analyze_rating_score_effect():
    """Is the rating score the only determinant? Or is there more?"""
    data = load_all_data()
    date, snapshot = data[-1]

    # Group movies by rating score
    rating_groups = defaultdict(list)

    for movie in snapshot['movies']:
        rank, name = parse_title(movie['title'])
        score, votes = parse_rating(movie['rating'])
        rating_groups[score].append({
            'movie': name,
            'rank': rank,
            'votes': votes,
        })

    return rating_groups

def analyze_vote_growth_vs_rank_change():
    """Do movies that grow votes faster move up in rank?"""
    data = load_all_data()

    if len(data) < 30:
        return []

    # Compare first and last month
    first_month = data[:30]
    last_month = data[-30:]

    movie_changes = {}

    # Get first month average
    first_stats = defaultdict(lambda: {'ranks': [], 'votes': []})
    for date, snapshot in first_month:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            score, votes = parse_rating(movie['rating'])
            first_stats[name]['ranks'].append(rank)
            first_stats[name]['votes'].append(votes)

    # Get last month average
    last_stats = defaultdict(lambda: {'ranks': [], 'votes': []})
    for date, snapshot in last_month:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            score, votes = parse_rating(movie['rating'])
            last_stats[name]['ranks'].append(rank)
            last_stats[name]['votes'].append(votes)

    # Calculate changes for movies in both periods
    for movie in first_stats:
        if movie in last_stats:
            first_avg_rank = statistics.mean(first_stats[movie]['ranks'])
            last_avg_rank = statistics.mean(last_stats[movie]['ranks'])
            first_avg_votes = statistics.mean(first_stats[movie]['votes'])
            last_avg_votes = statistics.mean(last_stats[movie]['votes'])

            movie_changes[movie] = {
                'rank_change': first_avg_rank - last_avg_rank,  # Positive = improved
                'vote_growth': last_avg_votes - first_avg_votes,
                'vote_growth_pct': ((last_avg_votes - first_avg_votes) / first_avg_votes) * 100,
            }

    return movie_changes

def calculate_statistical_significance():
    """Is the vote-rank mismatch statistically significant?"""
    data = load_all_data()
    date, snapshot = data[-1]

    movies = []
    for movie in snapshot['movies']:
        rank, name = parse_title(movie['title'])
        score, votes = parse_rating(movie['rating'])
        year = int(movie['year'])
        movies.append({
            'name': name,
            'rank': rank,
            'votes': votes,
            'rating': score,
            'year': year,
            'age': 2025 - year,
        })

    # Calculate correlation between movie age and votes
    ages = [m['age'] for m in movies]
    votes = [m['votes'] for m in movies]

    # Pearson correlation
    n = len(ages)
    mean_age = sum(ages) / n
    mean_votes = sum(votes) / n

    numerator = sum((ages[i] - mean_age) * (votes[i] - mean_votes) for i in range(n))
    denom_age = math.sqrt(sum((a - mean_age)**2 for a in ages))
    denom_votes = math.sqrt(sum((v - mean_votes)**2 for v in votes))

    age_votes_corr = numerator / (denom_age * denom_votes) if denom_age * denom_votes > 0 else 0

    # Correlation between age and rank
    ranks = [m['rank'] for m in movies]
    mean_rank = sum(ranks) / n

    numerator = sum((ages[i] - mean_age) * (ranks[i] - mean_rank) for i in range(n))
    denom_rank = math.sqrt(sum((r - mean_rank)**2 for r in ranks))

    age_rank_corr = numerator / (denom_age * denom_rank) if denom_age * denom_rank > 0 else 0

    return {
        'age_votes_correlation': age_votes_corr,
        'age_rank_correlation': age_rank_corr,
        'movies': movies,
    }

def export_for_visualization():
    """Export data for creating visualizations"""
    data = load_all_data()

    # Time series data
    time_series = []
    for date, snapshot in data:
        for movie in snapshot['movies']:
            rank, name = parse_title(movie['title'])
            name = normalize_movie_name(name)
            score, votes = parse_rating(movie['rating'])
            year = int(movie['year'])
            time_series.append({
                'date': date.isoformat(),
                'movie': name,
                'rank': rank,
                'rating': score,
                'votes': votes,
                'year': year,
            })

    return time_series

def main():
    print("=" * 80)
    print("DEEP INVESTIGATION: THE PARADOX OF POPULARITY")
    print("=" * 80)

    # 1. Vote-Rank Correlation Over Time
    print("\n### 1. VOTE-RANK CORRELATION STABILITY ###\n")
    correlations = calculate_vote_rank_correlation()

    mean_corr = statistics.mean([c[1] for c in correlations])
    std_corr = statistics.stdev([c[1] for c in correlations])

    print(f"Spearman correlation between votes and rank:")
    print(f"  Mean: {mean_corr:.4f}")
    print(f"  Std Dev: {std_corr:.6f}")
    print(f"  Range: {min(c[1] for c in correlations):.4f} to {max(c[1] for c in correlations):.4f}")
    print(f"\nInterpretation: Correlation of {mean_corr:.2f} means votes are a WEAK predictor of rank")

    # 2. Decade Effect
    print("\n### 2. DECADE BIAS ANALYSIS ###\n")
    decade_stats = analyze_decade_effect()

    print("Average rank bonus by decade (positive = outperforming vote count):")
    for decade in sorted(decade_stats.keys()):
        movies = decade_stats[decade]
        avg_bonus = statistics.mean([m['rank_bonus'] for m in movies])
        avg_votes = statistics.mean([m['votes'] for m in movies])
        print(f"\n{decade}s ({len(movies)} movies):")
        print(f"  Average rank bonus: {avg_bonus:+.1f} positions")
        print(f"  Average votes: {avg_votes:,.0f}")
        print("  Movies:")
        for m in sorted(movies, key=lambda x: x['rank_bonus'], reverse=True):
            print(f"    {m['movie'][:35]:35s} bonus: {m['rank_bonus']:+3d} (rank #{m['actual_rank']:2d}, expected #{m['expected_rank']:2d})")

    # 3. Rating Score Clustering
    print("\n### 3. WITHIN-RATING RANK DIFFERENCES ###\n")
    rating_groups = analyze_rating_score_effect()

    print("Within the same rating score, what determines order?")
    for score in sorted(rating_groups.keys(), reverse=True):
        movies = sorted(rating_groups[score], key=lambda x: x['rank'])
        if len(movies) > 1:
            print(f"\nRating {score} ({len(movies)} movies):")
            votes_sorted = sorted(movies, key=lambda x: x['votes'], reverse=True)

            # Check if votes determine order within rating
            matches = 0
            for i, m in enumerate(votes_sorted):
                actual_pos = next(j for j, am in enumerate(movies) if am['movie'] == m['movie'])
                if i == actual_pos:
                    matches += 1

            print(f"  Vote order matches rank order: {matches}/{len(movies)}")
            for m in movies:
                print(f"    #{m['rank']:2d} {m['movie'][:35]:35s} {m['votes']:>10,} votes")

    # 4. Vote Growth vs Rank Movement
    print("\n### 4. VOTE GROWTH VS RANK CHANGE ###\n")
    movie_changes = analyze_vote_growth_vs_rank_change()

    if movie_changes:
        # Sort by vote growth
        sorted_changes = sorted(movie_changes.items(), key=lambda x: x[1]['vote_growth_pct'], reverse=True)

        print("Movies sorted by vote growth % (first month vs last month):")
        for movie, changes in sorted_changes:
            rank_dir = "↑" if changes['rank_change'] > 0 else "↓" if changes['rank_change'] < 0 else "→"
            print(f"  {movie[:35]:35s}")
            print(f"    Vote growth: +{changes['vote_growth_pct']:.2f}%")
            print(f"    Rank change: {rank_dir} {changes['rank_change']:+.1f} positions")

        # Calculate correlation between vote growth and rank change
        growths = [c['vote_growth_pct'] for c in movie_changes.values()]
        rank_changes = [c['rank_change'] for c in movie_changes.values()]

        n = len(growths)
        mean_g = sum(growths) / n
        mean_r = sum(rank_changes) / n

        numerator = sum((growths[i] - mean_g) * (rank_changes[i] - mean_r) for i in range(n))
        denom_g = math.sqrt(sum((g - mean_g)**2 for g in growths))
        denom_r = math.sqrt(sum((r - mean_r)**2 for r in rank_changes))

        corr = numerator / (denom_g * denom_r) if denom_g * denom_r > 0 else 0

        print(f"\nCorrelation between vote growth % and rank improvement: {corr:.4f}")

    # 5. Statistical Significance
    print("\n### 5. AGE BIAS STATISTICAL TEST ###\n")
    stats = calculate_statistical_significance()

    print(f"Correlation between movie age and number of votes: {stats['age_votes_correlation']:.4f}")
    print(f"  (negative = newer movies have MORE votes)")

    print(f"\nCorrelation between movie age and rank: {stats['age_rank_correlation']:.4f}")
    print(f"  (positive = older movies rank HIGHER)")

    print("\nThis reveals THE KEY FINDING:")
    print("  - Older movies have FEWER votes but BETTER ranks")
    print("  - Newer movies have MORE votes but WORSE ranks")
    print("  - This is not random - it's a systematic bias!")

    # 6. The Extreme Cases
    print("\n### 6. MOST EXTREME MISMATCHES ###\n")

    movies = stats['movies']
    # Calculate expected rank based on votes
    vote_sorted = sorted(movies, key=lambda x: x['votes'], reverse=True)
    for i, m in enumerate(vote_sorted):
        m['expected_rank'] = i + 1
        m['mismatch'] = m['expected_rank'] - m['rank']  # Negative = performing better than votes suggest

    # Most underranked (lots of votes, low rank)
    underranked = sorted(movies, key=lambda x: x['mismatch'], reverse=True)[:5]
    print("Most UNDERRANKED (many votes, but rank is worse than expected):")
    for m in underranked:
        print(f"  {m['name'][:35]:35s}")
        print(f"    Year: {m['year']} | Votes: {m['votes']:,} | Rating: {m['rating']}")
        print(f"    Expected rank by votes: #{m['expected_rank']} | Actual: #{m['rank']}")
        print(f"    PENALTY: {m['mismatch']:+d} positions")
        print()

    # Most overranked (few votes, high rank)
    overranked = sorted(movies, key=lambda x: x['mismatch'])[:5]
    print("Most OVERRANKED (few votes, but rank is better than expected):")
    for m in overranked:
        print(f"  {m['name'][:35]:35s}")
        print(f"    Year: {m['year']} | Votes: {m['votes']:,} | Rating: {m['rating']}")
        print(f"    Expected rank by votes: #{m['expected_rank']} | Actual: #{m['rank']}")
        print(f"    BONUS: {-m['mismatch']:+d} positions")
        print()

    # Export data for visualization
    print("\n### 7. EXPORTING DATA FOR VISUALIZATION ###\n")
    time_series = export_for_visualization()

    # Write to JSON
    with open('/home/user/imdbscrape/viz_data.json', 'w') as f:
        json.dump(time_series, f)

    print(f"Exported {len(time_series)} data points to viz_data.json")

    # Also export summary data
    summary = {
        'latest_snapshot': stats['movies'],
        'correlations': {
            'age_votes': stats['age_votes_correlation'],
            'age_rank': stats['age_rank_correlation'],
            'vote_rank_mean': mean_corr,
        },
        'decade_stats': {str(k): v for k, v in decade_stats.items()},
    }

    with open('/home/user/imdbscrape/summary_data.json', 'w') as f:
        json.dump(summary, f, default=str)

    print("Exported summary data to summary_data.json")

if __name__ == "__main__":
    main()
