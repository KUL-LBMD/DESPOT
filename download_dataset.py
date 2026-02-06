import hashlib
import json
import logging
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import logging
import datetime
import os
import urllib3

DATAVERSE_URL = ""
API_KEY = ""
DOI = ""
VERSION = ""
OUTPUT_DIR = "./datasets"
LOG_PATH = ''

# Extends script by Jonathan Dan
__email__ = "rdm at kuleuven.be"
__license__ = "CC BY-SA 4.0 - https://creativecommons.org/licenses/by-sa/4.0/"



def config_logger():
    global LOG_PATH
    log_format = logging.Formatter(fmt='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

    filename = DOI.split('/')[-1] 
    LOG_PATH = os.path.join(
        '.', 
        f'download-{filename}-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.log'
        )
    file_handler = logging.FileHandler(filename=LOG_PATH, mode='a')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def fix_doi(doi):
    if 'doi.org' in doi:
        doi = 'doi:' + '/'.join(doi.split('/')[-2:])
    return doi

def fix_version(version):
    try:
        return str(float(version))
    except:
        logging.error(f'Invalid version :{version}')


def get_args():
    global DATAVERSE_URL
    global API_KEY
    global DOI 
    global VERSION
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser._optionals.title = 'arguments'
    parser.add_argument('-o', '--output', required=False, help='the output directory.')
    parser.add_argument('--host', required=True, help='Dataverse URL [REQUIRED]')
    parser.add_argument('--key', required=False,  help='Dataverse API token')
    parser.add_argument('--dataset', required=True,  help='dataset DOI in the format of  doi:......../...... or https://doi.org/......../......, [REQUIRED]')
    parser.add_argument('--version', required=True,  help='dataset version number [REQUIRED]')

    args = parser.parse_args()
    DATAVERSE_URL = args.host
    API_KEY = args.key
    DOI = fix_doi(args.dataset) 
    VERSION = fix_version(args.version)
    OUTPUT_DIR = args.output or os.path.join(*OUTPUT_DIR.split('/'), DOI.replace('doi:', 'doi-').replace('/', '-'), f'v{VERSION}')


urllib3.disable_warnings()
if __name__=='__main__':
    get_args()
    config_logger()
    logging.info(f'Dataverse: {DATAVERSE_URL}')
    logging.info(f'Dataset: {DOI}')
    logging.info(f'Version: {VERSION}')
    logging.info(f'Output folder: {OUTPUT_DIR}')
    print(f'Dataverse: {DATAVERSE_URL}')
    print(f'Dataset: {DOI}')
    print(f'Version: {VERSION}')
    print(f'Output folder: {OUTPUT_DIR}')
    print(f'Downloading dataset...')

    s = requests.Session()
    retries = Retry(total=5,
                backoff_factor=1)
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))

    # Get list of FILES
    url = DATAVERSE_URL + '/api/datasets/:persistentId/versions/' + VERSION + '/files'
    respFileList = s.get(url, params={'persistentId': DOI}, verify=False)
    failed_files = []
    restricted_files = []
    hash_difference = []
    downloaded = []
    present = []
    if respFileList.ok:
        fileList = json.loads(respFileList.content)
        
        # Download each file
        files = fileList['data']
        total=len(files)
        for i, file in enumerate(files):
            # Create directory structure
            file_path = OUTPUT_DIR.split('/') + file.get('directoryLabel', '').split('/') 
            filename = file['dataFile']['filename']
            description = f"{'/'.join([file.get('directoryLabel'),filename]) if file.get('directoryLabel') else filename}"
            filename = os.path.join(*(file_path + [filename]))
            logging.info('Downloading file: ' + description + f"({i+1}/{total})")
            print('Downloading file: ' + description + f"({i+1}/{total})")
            Path(os.path.join(*file_path)).mkdir(parents=True, exist_ok=True)

            # if file exists skip file:
            if os.path.exists(filename):
                md5Hash = hashlib.md5(open(filename, 'rb').read()).hexdigest()
                if md5Hash == file['dataFile']['md5']:
                    logging.info('Skipping already dowloaded file {}'.format(filename))
                    present.append(file['dataFile']['md5'])
                    downloaded.append(file['dataFile']['md5'])
                    continue

            # Download file
            try:
                url = DATAVERSE_URL + '/api/access/datafile/' + str(file['dataFile']['id'])
                if API_KEY:
                    headers = {'X-Dataverse-key': API_KEY}
                    respFile = s.get(url, headers=headers)
                else:
                    respFile = s.get(url)
            except Exception as e:
                logging.error('{} - {}'.format(file['dataFile']['id'], e))
            if respFile.ok:
                # Chec md5 hash
                md5Hash = hashlib.md5(respFile.content).hexdigest()
                if md5Hash == file['dataFile']['md5']:
                    # Write file to disk
                    with open(filename, 'wb') as edfFile:
                        edfFile.write(respFile.content)
                    downloaded.append(file['dataFile']['md5'])
                else:
                    logging.error('{} - MD5 hash difference'.format(file['dataFile']['id']))
            elif respFile.status_code == 403:
                failed_files.append(file['dataFile']['filename'])
                restricted_files.append(file['dataFile']['filename'])
                logging.error(f"{file['dataFile']['id']} - Failed to download file. {respFile.status_code}: access to file is restricted.")
            else:
                failed_files.append(file['dataFile']['filename'])
                try:
                    message =  json.loads(respFile.text).get('message')
                except:
                    message = respFile.text
                logging.error(f"{file['dataFile']['id']} - Failed to download file. {respFile.status_code}: {message}")
        logging.warning(f"{len(present)} of {len(files)} files were already present.")
        logging.warning(f"{len(failed_files)} of {len(files)} files failed.")
        logging.warning(f"{len(restricted_files)} skipped due to restricted access.")
        logging.info(f"{len(downloaded)} of {len(files)} files downloaded correctly")
        if failed_files:
            print(f"{len(failed_files)} downloads failed.")
        if restricted_files:
            print(f"{len(restricted_files)} were skipped due to restricted access.")
        print(f"{len(downloaded)} of {len(files)} files downloaded correctly.")
        print(f'See log file {LOG_PATH} for details.')
    else:
        try:
            message =  json.loads(respFileList.text).get('message')
        except:
            message = respFileList.text
            
        logging.error(f'Failed to retrieve dataset file list. {respFileList.status_code}: {message}')
