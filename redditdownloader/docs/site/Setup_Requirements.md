# Setup:
+ **[Get Python Installed.](https://www.python.org/downloads/)** - Typically the latest version will do. 
(See [Supported Versions](#supported-python-versions))
+ **Download:** Download this program, either using git or by [clicking here](https://github.com/shadowmoose/RedditDownloader/releases/latest) *(Latest Release)*. If you download the zip, unpack it.
+ **Install dependencies:** launch a terminal inside wherever you saved the program folder, and run the line:

```pip install -r requirements.txt```

*Windows Note: You can open a terminal in the folder by holding shift and right-clicking the folder (not the files inside it), then selecting "Open Window Here"*


+ Once the install finishes, launch the program with:

```python main.py```

*On first launch, it will run an assistant to aid in the setup process.*

# Creating Your Client Information:
Reddit requires that you have an authenticated app key to use their API. For this reason, RMD will prompt you during setup for a Client ID and Secret.

You'll be prompted for all of this by the Wizard, but this section will explain how to get that information anyways.

+ First, navigate to: [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
+ **Sign in on the same account you'll be scraping with** & click "Create and App" at the bottom of the page.
+ Name it whatever you'd like, so you'll remember what it's created for.
+ Select "Script" from the options.
+ Enter any description to remind yourself what this is for in the future.
+ Enter "http://localhost" in the "about" & "Redirect" URL fields.

*Your configuration should look something like this:*
![Settings](https://i.imgur.com/jy9Ok7v.png)

+ Click "Create App".

+ Now that the app has been created, scroll to it on the page and click "Edit"
+ From the menu it expands, copy the client ID (**Not the Name you gave it**) and client Secret.
![Demo](https://i.imgur.com/c6IxGmv.png)

+ When prompted by the Setup Wizard, enter these for the Client information!

## Notes:
* The Client ID & Secret you use **MUST** be generated by the same user that you provide username/password for.

* Whenever desired, you can automatically update the program and its dependencies by running:

```python main.py --update```

* ...or by manually downloading the latest release and re-running:

```pip install -r requirements.txt```

See [Here](../Arguments.md) for more information on (optional) parameters, for advanced use & automation.

See [User-Guide](./User_Guide.md) for examples on running now that it's been set up.

## Supported Python Versions:
You should be fine with 3.4 up, and maybe even slightly earlier, but you can view the only versions I officially support at [The Travis Build Page](https://travis-ci.org/shadowmoose/RedditDownloader). It automatically checks the most recent commit, and runs through a strict set of tests to make sure nothing's broken.
