# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  print('maybe_download:  ' + filename)
  """Download a file if not present, and make sure it's the right size."""
  url = 'http://mattmahoney.net/dc/'
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
