import os
from ..objects.mobject import mobject
from ..objects import plotobjects
from ..utils.misc import iterate, iterate_dicts
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader , PdfWriter
from datetime import datetime

class plot(mobject):
    """
    `plot` is a command used to make plots or animations for the `plot` or `pdf` objects defined in the configuration file.

    Example:
    >>> chump.exe plot example.toml
    >>> chump.exe plot example.toml --only Hydrograph
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.dpi = self.dict.get('dpi', 200)
        """`dpi` set the default dpi for the PDF
        """

        self.pages = []
        """`pages` or keys starting with 'pages' set plot objects

        Example:
        >>> pages  = {tsplot='plotStreamflow'}
        >>> pages  = {tsplot=['plotStreamflowPEST', 'plotHydrograph'], {mapplot='plotGwMap'}}
        >>> pages2 = {scatterplot='ObsSim'}
        """
        for k, v in self.dict.items():
            if k.startswith('pages'):
                if isinstance(v, dict):
                    self.pages.append(v)
                elif isinstance(v, list):
                    self.pages += v
                else:
                    raise ValueError(f'Unknown plot type in {self.fullname}')

        self.metadata = self.dict.get('metadata', self.parent.metadata)
        """`metadata` set the metadata of the PDF

        Example:
        >>> metadata = {title='SSPA Project 111',
        >>>             author='SSPAmodler',
        >>>             subject='Groundwater model results'}
        """

    def run(self, ):
        """@private"""

        # check if no page to add
        hasContent = False
        for p in iterate_dicts(self.pages):
            if len(p) > 0:
                hasContent = True

        if not hasContent:
            return

        print(f'Creating PDF {self.name} ...')

        bookmarks = {}

        with PdfPages(self.name + '_.pdf') as pdf:
            for pages in self.pages:
                for k, kv in pages.items():
                    for v in iterate(kv):
                        p = getattr(plotobjects, k)(v, self.parent)
                        p.initialize_plot()
                        bookmarks[v] = p.write_pdf(pdf, self.dpi, False)


        # write bookmarks
        # print(bookmarks)
        output = PdfWriter() # open an output object
        meta = {'/'+k.capitalize():v for k, v in self.metadata.items()}
        meta['/Created'] = datetime.now().strftime('%Y-%m-%d')
        meta['/Title'] = meta.get('/Title', 'Produced using CHUMP, a SSPA Product, All Rights Reserved')
        output.add_metadata(meta)
        with open(self.name+'_.pdf', 'rb') as fr:
            input = PdfReader (fr) # read PDF

            for i in range(len(input.pages) ):
                output.add_page(input.pages[i]) # insert page

            def writebookmark(bdict, ipage=0, parent=None):
                if bdict is None:
                    return ipage + 1
                if isinstance(bdict, dict):
                    for k, v in bdict.items():
                        newparent = output.add_outline_item(k, ipage, parent)
                        ipage = writebookmark(v, ipage, newparent)
                    return ipage

                elif isinstance(bdict, list):
                    for b in bdict:
                        ipage = writebookmark(b, ipage, parent)
                    return ipage
                else:
                    output.add_outline_item(bdict, ipage, parent=parent)
                    return ipage + 1

            writebookmark(bookmarks,)

            output.page_mode  = "/UseOutlines" #This is what tells the PDF to open to bookmarks
            with open(self.name+'.pdf','wb') as fw: #creating result pdf
                output.write(fw) #writing to result pdf

        os.remove(self.name+'_.pdf')
        if self.verbose > 0:
            print(f' {self.name}.pdf written successfully.\n')
