# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from geventhttpclient import HTTPClient
from geventhttpclient.url import URL
import rapidjson as json
import numpy as np

from tritonhttpclient.utils import *


def raise_if_error(response):
    """
    Raise InferenceServerException if received non-Success
    response from the server
    """
    if response.status_code != 200:
        error_response = json.loads(response.read())
        raise_error(error_response["error"])


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using http protocol.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8000'.

    connection_count : int
        The number of connections to create for this client.
        Default value is 1.

    connection_timeout : float
        The timeout value for the connection. Default value
        is 60.0 sec.
    
    network_timeout : float
        The timeout value for the network. Default value is
        60.0 sec

    verbose : bool
        If True generate verbose output. Default value is False.

    Raises
        ------
        Exception
            If unable to create a client.

    """

    def __init__(self,
                 url,
                 connection_count=1,
                 connection_timeout=60.0,
                 network_timeout=60.0,
                 verbose=False):
        self._last_request_id = None
        self._parsed_url = URL("http://" + url)
        self._client_stub = HTTPClient.from_url(
            self._parsed_url,
            concurrency=connection_count,
            connection_timeout=connection_timeout,
            network_timeout=network_timeout)
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close the client. Any future calls to server
        will result in an Error.

        """
        self._client_stub.close()

    def is_server_live(self, headers={}):
        """Contact the inference server and get liveness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying the headers that must be
            sent alongwith the request.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        Exception
            If unable to get liveness.

        """
        response = self._client_stub.get("v2/health/live", headers=headers)
        return response.status_code == 200

    def is_server_ready(self, headers={}):
        """Contact the inference server and get readiness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying the headers that must be
            sent alongwith the request.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        Exception
            If unable to get readiness.

        """
        response = self._client_stub.get("v2/health/ready", headers=headers)
        return response.status_code == 200

    def is_model_ready(self, model_name, model_version="", headers={}):
        """Contact the inference server and get the readiness of specified model.

        Parameters
        ----------
        model_name: str
            The name of the model to check for readiness.
        model_version: str
            The version of the model to check for readiness. The default value
            is an empty string which means then the server will choose a version
            based on the model and internal policy.
        headers: dict
            Optional dictionary specifying the headers that must be
            sent alongwith the request.

        Returns
        -------
        bool
            True if the model is ready, False if not ready.

        Raises
        ------
        Exception
            If unable to get model readiness.

        """
        if not model_version:
            request_uri = "v2/models/{}/ready".format(model_name)
        else:
            request_uri = "v2/models/{}/versions/{}/ready".format(
                model_name, model_version)
        response = self._client_stub.get(request_uri, headers=headers)
        return response.status_code == 200

    def get_server_metadata(self, headers={}):
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying the headers that must be
            sent alongwith the request.

        Returns
        -------
        dict
            The JSON dict holding the metadata.

        Raises
        ------
        Exception
            If unable to get server metadata.

        """
        response = self._client_stub.get("v2", headers=headers)
        raise_if_error(response)
        metadata = json.loads(response.read())
        return metadata

    def get_model_metadata(self, model_name, model_version="", headers={}):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get metadata. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying the headers that must be
            sent alongwith the request

        Returns
        -------
        dict
            The JSON dict holding the metadata.

        Raises
        ------
        Exception
            If unable to get model metadata.

        """
        if not model_version:
            request_uri = "v2/models/{}".format(model_name)
        else:
            request_uri = "v2/models/{}/versions/{}".format(
                model_name, model_version)
        response = self._client_stub.get(request_uri, headers=headers)
        raise_if_error(response)
        metadata = json.loads(response.read())
        return metadata

    def infer(self,
              inputs,
              model_name,
              model_version="",
              outputs=None,
              request_id=None,
              parameters=None,
              headers={}):
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        outputs : list
            A list of InferOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        model_name: str
            The name of the model to run inference.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        parameters: dict
            Optional inference parameters described as key-value pairs.
        headers: dict
            Optional dictionary specifying the headers that must be
            sent alongwith the request

        Returns
        -------
        InferResult
            The object holding the result of the inference, including the
            statistics.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """
        infer_request = {}
        if request_id:
            infer_request['id'] = request_id
        if parameters:
            infer_request['parameters'] = parameters
        infer_request['inputs'] = [
            this_input._get_tensor() for this_input in inputs
        ]
        if outputs:
            infer_request['outputs'] = [
                this_output._get_tensor() for this_output in outputs
            ]

        request_body = json.dumps(infer_request)
        if not model_version:
            request_uri = "v2/models/{}/infer".format(model_name)
        else:
            request_uri = "v2/models/{}/versions/{}/infer".format(
                model_name, model_version)
        response = self._client_stub.post(request_uri=request_uri,
                                          body=request_body,
                                          headers=headers)
        response = self._client_stub.get(request_uri, headers=headers)
        result = InferResult(response.read())

        return result

class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object

    """

    def __init__(self, name):
        self._input = {}
        self._parameters = {}
        self._input['name'] = name

    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._input['name']

    @property
    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        self._input['datatype']

    @datatype.setter
    def datatype(self, value):
        """Sets the datatype for the input associated with this
        object

        Parameters
        ----------
        value : str
            The datatype of input
        """
        self._input['datatype'] = value

    @property
    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """
        self._input['shape']

    @shape.setter
    def shape(self, value):
        """Sets the shape of input associated with this object.

        Parameters
        ----------
        value : list
            The shape of input
        """
        self._input['shape'] = value

    def set_data_from_numpy(self, input_tensor):
        """Set the tensor data (datatype, shape, contents) from the
        specified numpy array for input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        self._input['datatype'] = np_to_triton_dtype(input_tensor.dtype)
        self._input['shape'] = input_tensor.shape
        # FIXMEV2 Use Binary data when available on the server.
        self._input['data'] = [val.item() for val in input_tensor.flatten()]

    def set_parameter(self, key, value):
        """Adds the specified key-value pair in the requested input parameters

        Parameters
        ----------
        key : str
            The name of the parameter to be included in the request. 
        value : str/int/bool
            The value of the parameter
        
        """
        if not type(key) is str:
            raise_error(
                "only string data type for key is supported in parameters")

        self._parameters[key] = value

    def clear_parameters(self):
        """Clears all the parameters that have been added to the request.
        
        """
        self._parameters.clear()

    def _get_tensor(self):
        """Retrieve the underlying input as json string.

        Returns
        -------
        str
            The underlying tensor as serialized json string
        """
        self._input['parameters'] = self._parameters
        return self._input


class InferOutput:
    """An object of InferOutput class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object
    """

    def __init__(self, name):
        self._output = {}
        self._parameters = {}
        self._output['name'] = name

    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._output['name']

    def set_parameter(self, key, value):
        """Adds the specified key-value pair in the requested output parameters

        Parameters
        ----------
        key : str
            The name of the parameter to be included in the request. 
        value : str/int/bool
            The value of the parameter
        
        """
        if not type(key) is str:
            raise_error(
                "only string data type for key is supported in parameters")

        self._parameters[key] = value
    
    def clear_parameters(self):
        """Clears all the parameters that have been added to the request.
        
        """
        self._parameters.clear()

    def _get_tensor(self):
        """Retrieve the underlying input as json string.

        Returns
        -------
        str
            The underlying tensor as serialized json string
        """
        self._output['parameters'] = self._parameters
        return self._output


class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : dict
        The inference response from the server
    """

    def __init__(self, result):
        self._result = json.loads(result)

    def as_numpy(self, name):
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.
    
        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        for output in self._result['outputs']:
            if output['name'] == name:
                datatype = output['datatype']
                # FIXME: Add the support for binary data when available with server
                np_array = np.array(output['data'], dtype=triton_to_np_dtype(datatype))
                resize(np_array, output['shape'])
                return np_array
        return None

    def get_response(self):
        """Retrieves the complete response

        Returns
        -------
        dict
            The underlying response dict.
        """
        return self._result
