def test(model_path: str, config_path: str, ortho_path="test.tif"):
    """
    Download test.tif example orthophotoplane if it is not present and no other orthophotoplane is passed and run
    border cars detection with chosen model and config. Save bounding boxes list as a text file and a copy of
    orthophotoplane with drawn bounding boxes to results folder in your project directory
    :param model_path: path to .pth model file
    :param config_path: path to .py config file
    :param ortho_path: path to orthophotoplane in TIFF image format
    """
    if not (ortho_path.endswith(".tif") or ortho_path.endswith(".tiff")):
        raise ValueError(ortho_path + " is not in TIFF image format")

    import os

    if not os.path.exists(ortho_path):
        if ortho_path is "test.tif":
            print("test.tif not found, downloading example (Yandex Disk, 3.5 GB)")

            import requests
            from urllib.parse import urlencode

            response = requests.get(('https://cloud-api.yandex.net/v1/disk/public/resources/download?' +
                                     urlencode(dict(public_key="https://disk.yandex.ru/d/qnCkCGtJVDSJcQ"))))
            url = response.json()['href']
            download = requests.get(url)

            with open(ortho_path, 'wb') as ortho:
                ortho.write(download.content)

            print("test.tif downloaded, proceeding")
        else:
            raise FileNotFoundError(str(ortho_path) + " not found")
    else:
        print(ortho_path + " found, proceeding")

    from bordercars.detector import Detector

    detector = Detector(model_path, config_path)
    bboxes = detector(ortho_path)

    if not os.path.exists("results"):
        os.makedirs("results")

    detector.save(bboxes, "results/" + model_path[model_path.rfind('/') + 1:model_path.rfind('.')] + ".txt")
    detector.draw(bboxes, ortho_path, "results/" + model_path[model_path.rfind('/') + 1:model_path.rfind('.')] + ".tif")
