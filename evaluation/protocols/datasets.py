"""Scene lists shared by dataset optimization launchers and evaluation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    data_name: str
    kind: str
    iteration: int
    train_root: str
    scenes: tuple[str, ...]


DATASETS = {
    'DAVIS': Dataset(
        data_name='DAVIS_processed', kind='real', iteration=20000,
        train_root='DAVIS_processed',
        scenes=(
            'bear',
            'blackswan',
            'camel',
            'cows',
            'goat',
            'impala0',
            'mallard-water',
            'pigs_38',
        ),
    ),
    'APTv2': Dataset(
        data_name='APTv2', kind='real', iteration=10000,
        train_root='APTv2',
        scenes=(
            '14elephant/video20_clip10_frame',
            '14elephant/video26_clip2_frame',
            '14elephant/video2_clip3_frame',
            '15hippo/video19_clip2_frame-done',
            '15hippo/video22_clip4_frame-34',
            '15hippo/video5_clip1_frame-done',
            '16raccoon/video15_clip6_frame-35',
            '17rhino/video10_clip2_frame',
            '17rhino/video11_clip6_frame',
            '17rhino/video12_clip1_frame',
            '17rhino/video2_clip2_frame',
            '17rhino/video3_clip6_frame',
            '17rhino/video4_clip13_frame',
            '17rhino/video4_clip14_frame',
            '17rhino/video4_clip18_frame',
            '18giraffe/video14_clip2_frame',
            '18giraffe/video6_clip7_frame',
            '19tiger/video12_clip2_0m27s-0m30s_frame',
            '19tiger/video12_clip5_1m3s-1m5s_frame',
            '19tiger/video12_clip9_2m58s-3m0s_frame',
            '19tiger/video3_clip43_12m42s-12m45s_frame',
            '1deer/clip22',
            '1deer/clip3',
            '1deer/clip5',
            '20lion/video4_clip46_45m4s-45m6s_frame',
            '20lion/video8_clip2_0m20s-0m22s_frame',
            '20lion/video8_clip3_0m26s-0m29s_frame',
            '21panda/video2_clip19_7m45s-7m48s_frame',
            '21panda/video2_clip8_1m51s-1m53s_frame',
            '22cheetah/video12_clip3_0m54s-0m57s_frame',
            '22cheetah/video1_clip2_1m45s-1m47s_frame',
            '22cheetah/video7_clip14_3m51s-3m54s_frame',
            '23black-bear/video11_clip22_22m13s-22m15s_frame',
            '23black-bear/video11_clip8_7m39s-7m41s_frame',
            '23black-bear/video2_clip1_0m21s-0m23s_frame',
            '23black-bear/video2_clip13_8m31s-8m34s_frame',
            '23black-bear/video6_clip11_3m35s-3m38s_frame',
            '23black-bear/video6_clip7_2m18s-2m21s_frame',
            '24polar-bear/video1_clip41_8m21s-8m23s_frame',
            '24polar-bear/video1_clip54_11m16s-11m18s_frame',
            '24polar-bear/video2_clip12_3m42s-3m44s_frame',
            '24polar-bear/video4_clip9_6m16s-6m21s',
            '25antelope/video51_14m38s-14m46s',
            '26fox/video34_clip2_11m46s-11m48s_frame',
            '26fox/video60_clip1_0m1s-0m3s_frame',
            '26fox/video61_clip4_1m27s-1m29s_frame',
            '26fox/video61_clip5_9m4s-9m6s_frame',
            '27buffalo/video28_clip2_0m15s-0m17s_frame',
            '27buffalo/video35_clip7_5m2s-5m4s_frame',
            '27buffalo/video40_clip6_3m2s-3m4s_frame',
            '27buffalo/video9_clip4_3m14s-3m17s_frame',
            '29wolf/video14_clip5_2m37s-2m39s_frame',
            '29wolf/video30_clip2_5m58s-6m0s_frame',
            '29wolf/video6_clip3_1m24s-1m26s_frame',
            '2dog/v12c1',
            '2dog/v16c3',
            '2dog/v25c2',
            '30sheep/video38_clip1_0m21s-0m23s_frame-OK',
            '3cat/v14c5',
            '3cat/v21c2',
            '3cat/v7c2',
            '4horse/v0C5',
            '4horse/v2c0',
            '4horse/video21_clip2_0m4s-0m6s_frame',
            '5rabbit/v25c1',
            '5rabbit/v6c2',
            '6pig/v1c2',
            '6pig/v1c8',
        ),
    ),
    'Artemis': Dataset(
        data_name='Artemis_processed', kind='artemis', iteration=20000,
        train_root='Artemis_processed/train',
        scenes=(
            'beagle_dog/s1_24fps',
            'bear/run',
            'bear/walk',
            'cat/run',
            'cat/walk_final',
            'cat/walkprogressive_noz',
            'cat/walksniff',
            'duck/eat_grass',
            'duck/swim',
            'duck/walk',
            'fox/attitude',
            'fox/run',
            'fox/walk',
            'lion/Run',
            'lion/Walk',
            'panda/acting',
            'panda/run',
            'panda/walk',
            'whiteTiger/roaringwalk',
            'whiteTiger/run',
            'wolf/Damage',
            'wolf/Howling',
            'wolf/Run',
            'wolf/Walk',
        ),
    ),
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    args = parser.parse_args()
    print('\n'.join(DATASETS[args.dataset].scenes))
