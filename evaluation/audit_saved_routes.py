"""Analyze saved Wiki Game routes using only Python's standard library."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze_routes(baseline, articles):
    episodes = baseline.get('episodes')
    settings = baseline.get('settings', {})
    budget = settings.get('max_clicks')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('Baseline needs a nonempty episodes list.')
    if type(budget) is not int or budget < 1:
        raise ValueError('Baseline settings must contain a positive max_clicks.')
    if not isinstance(articles, dict) or not articles:
        raise ValueError('Article graph must be a nonempty dictionary.')
    results = []
    for episode in episodes:
        start, target = episode['start'], episode['target']
        path = episode['path']
        clicks = episode['clicks']
        label = f'{start} -> {target}'
        if not isinstance(path, list) or not path or path[0] != start or path[-1] not in articles:
            raise ValueError(f'{label}: invalid path endpoints.')
        if type(clicks) is not int or clicks < 1 or clicks > budget or clicks != len(path)-1:
            raise ValueError(f'{label}: clicks/path length/budget disagree.')
        if start not in articles or target not in articles or start == target:
            raise ValueError(f'{label}: unknown or identical start and target.')
        if target in path[:-1]:
            raise ValueError(f'{label}: recorded moves after reaching the target.')
        success = path[-1] == target
        if episode.get('success') is not success:
            raise ValueError(f'{label}: success flag disagrees with the final node.')
        expected_return = 1 + .1*(budget-clicks)/budget if success else 0.0
        recorded_return = episode.get('episode_return')
        if (not isinstance(recorded_return, (int, float)) or not math.isfinite(recorded_return)
                or not math.isclose(recorded_return, expected_return, abs_tol=1e-8, rel_tol=0)):
            raise ValueError(f'{label}: reward disagrees with the configured reward formula.')

        missed, available, revisits, self_links, backtracks = [], [], [], [], []
        visited = {start}
        for index, (current, chosen) in enumerate(zip(path, path[1:])):
            if current not in articles or chosen not in articles:
                raise ValueError(f'{label}: unknown article at step {index+1}.')
            outgoing = articles[current]['outgoing']
            if chosen not in outgoing:
                raise ValueError(f'{label}: step {index+1} is not a valid hyperlink: {current} -> {chosen}')
            selected_id = outgoing.index(chosen)
            if target in outgoing:
                event = {'step': index+1, 'remaining_clicks_before_move': budget-index,
                         'current_id': current, 'current_title': articles[current]['title'],
                         'target_id': target, 'target_title': articles[target]['title'],
                         'expected_action_id': outgoing.index(target),
                         'selected_action_id': selected_id,
                         'selected_article_id': chosen, 'selected_title': articles[chosen]['title'],
                         'link_count': len(outgoing), 'selected_target': chosen == target}
                available.append(event)
                if chosen != target:
                    missed.append(event)
            if chosen in visited:
                revisits.append(index+1)
            if chosen == current:
                self_links.append(index+1)
            if index >= 1 and chosen == path[index-1] and chosen != current:
                backtracks.append(index+1)
            visited.add(chosen)

        visits = Counter(path)
        results.append({
            'start': start, 'start_title': articles[start]['title'],
            'target': target, 'target_title': articles[target]['title'],
            'success': success, 'clicks': clicks, 'episode_return': recorded_return,
            'reason': episode.get('reason'), 'path': path,
            'direct_target_opportunities': len(available),
            'missed_direct_target_count': len(missed),
            'unique_articles_with_missed_target': len({e['current_id'] for e in missed}),
            'missed_direct_target_steps': missed,
            'target_clicked_steps': [e['step'] for e in available if e['selected_target']],
            'revisit_count': len(revisits), 'revisit_steps': revisits,
            'self_link_count': len(self_links), 'self_link_steps': self_links,
            'immediate_backtrack_count': len(backtracks), 'immediate_backtrack_steps': backtracks,
            'most_visited': [{'article_id': a, 'title': articles[a]['title'], 'visits': n}
                             for a, n in visits.most_common(4)],
        })

    failures = [e for e in results if not e['success']]
    opportunities = sum(e['direct_target_opportunities'] for e in results)
    clicked = sum(len(e['target_clicked_steps']) for e in results)
    summary = {
        'episodes': len(results), 'successes': len(results)-len(failures), 'failures': len(failures),
        'total_clicks': sum(e['clicks'] for e in results),
        'all_recorded_transitions_and_returns_valid': True,
        'direct_target_opportunities': opportunities,
        'target_clicked_count': clicked,
        'missed_direct_target_count': sum(e['missed_direct_target_count'] for e in results),
        'direct_target_selection_rate_percent': 100*clicked/opportunities if opportunities else None,
        'episodes_with_missed_direct_targets': sum(bool(e['missed_direct_target_count']) for e in results),
        'failed_episodes_with_missed_direct_targets': sum(bool(e['missed_direct_target_count']) for e in failures),
        'failed_episodes_without_direct_target_opportunity': sum(e['direct_target_opportunities'] == 0 for e in failures),
        'episodes_with_revisits': sum(bool(e['revisit_count']) for e in results),
        'failed_episodes_with_revisits': sum(bool(e['revisit_count']) for e in failures),
        'total_revisits': sum(e['revisit_count'] for e in results),
        'self_link_moves': sum(e['self_link_count'] for e in results),
        'immediate_backtracks': sum(e['immediate_backtrack_count'] for e in results),
        'termination_reasons': dict(Counter(e['reason'] for e in results)),
    }
    return {'summary': summary, 'baseline_settings': settings, 'episodes': results,
            'interpretation': {
                'missed_direct_target': 'Target was in the current outgoing links, but a different valid link was selected. This alone cannot distinguish ID copying from deliberate article selection.',
                'revisit': 'A move to any previously visited node; a self-link is included. Revisits are legal and may sometimes be useful.',
                'immediate_backtrack': 'A distinct-node A -> B -> A pattern, counted at the return move.',
                'overlap': 'Missed-target, revisit, self-link and backtrack counts can overlap; do not add them as disjoint causes.',
                'opportunities': 'Repeated visits with the target available count as repeated opportunities, not independent samples.',
                'no_direct_opportunity': 'The route never reached a node with a direct link to the target; this does not identify the reason for each earlier choice.',
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'evaluation/results/baseline_colon_4096.json')
    parser.add_argument('--articles', type=Path, default=ROOT/'data/processed/articles.json')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/results/route_audit_colon_4096.json')
    args = parser.parse_args()
    if args.output.resolve() in {args.baseline.resolve(), args.articles.resolve()}:
        parser.error('Output must not overwrite either input file.')
    try:
        baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
        articles = json.loads(args.articles.read_text(encoding='utf-8'))
        result = analyze_routes(baseline, articles)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f'Route audit failed: {error}\n')
    result['inputs'] = {'baseline': str(args.baseline), 'articles': str(args.articles)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
    for ep in result['episodes']:
        for event in ep['missed_direct_target_steps']:
            print(f"MISSED TARGET | {ep['start_title']} -> {ep['target_title']} | "
                  f"step {event['step']} at {event['current_title']}: "
                  f"expected {event['expected_action_id']}: {event['target_title']}, "
                  f"selected {event['selected_action_id']}: {event['selected_title']}")
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
