import pandas as pd
from sklearn.datasets import fetch_california_housing


def main() -> int:
    output_filename = 'california-housing.csv'

    print('Fetching California Housing data...', flush=True)
    california_housing = fetch_california_housing(as_frame=True)
    df = pd.DataFrame(california_housing.data)
    df = df.assign(MedianHouseValue=california_housing.target)

    print('Read {:,} rows x {:,} columns.'.format(*df.shape))
    df.to_csv(output_filename, index=False)
    print(f"Output data to file: '{output_filename}'")

    print('Description:')
    print(california_housing.DESCR)

    return 0


if __name__ == '__main__':
    exit(main())