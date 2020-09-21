## OpenDR Toolkit Documentation

### Show the Documentation

First download all the required dependencies by executing the following command in the project root folder
```
make documentation
```

Then, start a Python local server
```
python3 -m http.server
```

You can now open the documentation in your browser using the URL `localhost:8000/docs/index.html`.


### Write the Documentation

Each public function of the OpenDR Toolkit should be documented.
The technical documentation is located in the `docs/reference` directory.
It is written using the Markdown format and a script converts it to HTML format.
Please refer to existing pages to use a consistent format.

