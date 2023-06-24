import sys
import random
import numpy as np
import pandas as pd


def main():
    random.seed(10081995)
    games_played_outcomes = np.array([], dtype=int)
    num_epochs = 10000
    wr = 0.60
    lp_per_wl = 17
    avg_game_length = 29      # minutes
    start_lp = 156            # gold 3, 56 lp
    outcome_window_size = 34  # 34 == 1 stdev
    if (len(sys.argv) >= 2):
        try:
            wr = float(sys.argv[1])
            try:
                start_lp = int(sys.argv[2])
            except:
                pass
        except:
            print('Usage: python3 lol.py <wr> <lp>')
            sys.exit(1)
    if (wr <= 0.50):
        print('wr too low to climb')
        sys.exit(1)
    for _ in range(num_epochs):
        lp = start_lp
        goal_lp = 400  # plat 4 promos
        games_played = 0
        while lp < goal_lp:
            game_result = random.random() < wr
            lp = lp + lp_per_wl if game_result else lp - lp_per_wl
            games_played += 1
        games_played_outcomes = np.append(games_played_outcomes, games_played)
    games_played_outcomes = np.sort(games_played_outcomes)
    gpo_bincount = np.bincount(games_played_outcomes)
    gpo_indexes = np.where(gpo_bincount > 0)
    gpo_counts = gpo_bincount[gpo_indexes]
    res = np.vstack((gpo_indexes, gpo_counts)).T
    res[:, 1] = res[:, 1].cumsum().astype(np.float64) / num_epochs * 100
    res = res[np.where((res[:, 1] % 5 <= 1) &
                       (res[:, 1] > 50-outcome_window_size) &
                       (res[:, 1] <= 50+outcome_window_size))]
    res = np.c_[res, res[:, 0]*avg_game_length/60]
    columns_titles = ['GP', 'Prob (%)', 'Time (hrs)']
    df = pd.DataFrame(res, columns=columns_titles).astype(int)
    columns_titles[0], columns_titles[1] = columns_titles[1], columns_titles[0]
    df = df.reindex(columns=columns_titles)
    print(df.to_string(index=False, justify='center'))


if __name__ == '__main__':
    main()
