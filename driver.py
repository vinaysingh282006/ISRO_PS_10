import os
import argparse
import subprocess
from utils.logging_utils import setup_logging
from utils.file_utils import find_file


def run_script(script_name, logger, *args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, 'scripts')
    script_path = os.path.join(scripts_dir, script_name)
    command = ['python', script_path] + list(args)
    logger.info(f"Running: {' '.join(command)}")
    try:
        # Add the project root to PYTHONPATH so scripts can import from utils
        env = os.environ.copy()
        env['PYTHONPATH'] = base_dir + os.pathsep + env.get('PYTHONPATH', '')
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, env=env)
        if result.stdout:
            logger.info(f"STDOUT from {script_name}:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR from {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise e


def main():

    parser = argparse.ArgumentParser(
        description='IR-Colorization Dataset Generation Baseline'
    )

    parser.parse_args()

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    input_root = os.path.join(
        base_dir,
        'input'
    )

    output_dir = os.path.join(
        base_dir,
        'output'
    )

    output_downscale_dir = os.path.join(
        output_dir,
        'downscaled_data'
    )

    output_rgb_dir = os.path.join(
        output_dir,
        'rgb_images'
    )

    output_patches_dir = os.path.join(
        output_dir,
        'patches'
    )

    os.makedirs(
        output_downscale_dir,
        exist_ok=True
    )

    os.makedirs(
        output_rgb_dir,
        exist_ok=True
    )

    os.makedirs(
        output_patches_dir,
        exist_ok=True
    )

    logger = setup_logging(
        output_dir
    )

    if not os.path.isdir(input_root):

        logger.error(
            f"Input root directory {input_root} not found."
        )

        return

    product_folders = sorted([
        e for e in os.listdir(input_root)
        if os.path.isdir(
            os.path.join(
                input_root,
                e
            )
        )
    ])

    logger.info(
        f"Found {len(product_folders)} products"
    )

    # ==========================================
    # PROCESS ALL PRODUCTS
    # ==========================================

    for product_id in product_folders:

        input_dir = os.path.join(
            input_root,
            product_id
        )

        logger.info(
            f"Processing {product_id}"
        )

        band2_path = find_file(
            input_dir,
            '_B2'
        )

        band3_path = find_file(
            input_dir,
            '_B3'
        )

        band4_path = find_file(
            input_dir,
            '_B4'
        )

        band10_path = find_file(
            input_dir,
            '_B10'
        )

        if not all([
            band2_path,
            band3_path,
            band4_path,
            band10_path
        ]):

            logger.warning(
                f"Skipping {product_id}: Missing required bands."
            )

            continue

        try:

            rgb_output_path = os.path.join(
                output_rgb_dir,
                f"{product_id}_rgb_30m.tif"
            )

            run_script(
                'merge_rgb.py',
                logger,
                band4_path,
                band3_path,
                band2_path,
                rgb_output_path
            )

            downscaled_rgb_100m = os.path.join(
                output_downscale_dir,
                f"{product_id}_rgb_100m.tif"
            )

            run_script(
                'downscale.py',
                logger,
                rgb_output_path,
                downscaled_rgb_100m,
                '3.33'
            )

            downscaled_tir_100m = os.path.join(
                output_downscale_dir,
                f"{product_id}_tir_100m.tif"
            )

            run_script(
                'downscale.py',
                logger,
                band10_path,
                downscaled_tir_100m,
                '3.33'
            )

            downscaled_tir_200m = os.path.join(
                output_downscale_dir,
                f"{product_id}_tir_200m.tif"
            )

            run_script(
                'downscale.py',
                logger,
                band10_path,
                downscaled_tir_200m,
                '6.67'
            )

            logger.info(
                f"Finished processing {product_id}"
            )

        except Exception as e:

            logger.error(
                f"Error processing {product_id}: {e}"
            )

    # ==========================================
    # CREATE PATCHES ONCE
    # ==========================================

    logger.info("=" * 60)
    logger.info("GENERATING PATCHES")
    logger.info("=" * 60)

    run_script(
        'create_patches.py',
        logger,
        '--input_dir',
        output_downscale_dir,
        '--output_dir',
        output_patches_dir
    )

    logger.info("=" * 60)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("=" * 60)

    logger.info(
        f"Patches saved to: {output_patches_dir}"
    )


if __name__ == '__main__':
    main()
