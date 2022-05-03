#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Fri Jan 17 14:29:05 2020
"""
import json, time, importlib.resources

class Namelist(object):
    def __init__(self):
        with importlib.resources.path("barotropicSWEs.Configuration",
                                      "defaults.json") as data_path:
            with open(data_path) as f:
                namelist = json.load(f)
        self.set_parameters(namelist)
        self.start_time = time.time()

    def set_parameters(self, namelist):
        avail = {}
        doc = {}
        for d in namelist.keys():
            dd = namelist[d]
            for name in dd.keys():
                val = dd[name]['default']
                setattr(self, name, val)
                if 'avail' in dd[name]:
                    avail[name] = dd[name]['avail']
                if 'doc' in dd[name]:
                    doc[name] = dd[name]['doc']
        self.avail = avail
        self.doc = doc

    def man(self, name):
        if name in self.doc:
            helpstr = self.doc[name]
            if name in self.avail:
                availstr = ', '.join([str(l) for l in self.avail[name]])
                helpstr += ' / available values = ['+availstr+']'
        else:
            helpstr = 'no manual for this parameter'
        print('Manual for %s: %s' % (name, helpstr))

    def manall(self):
        ps = self.listall()
        for p in ps:
            self.man(p)

    def checkall(self):
        for p, avail in self.avail.items():
            if getattr(self, p) in avail:
                # the parameter 'p' is well set
                pass
            else:
                msg = 'parameter "%s" should be in ' % p
                msg += str(avail)
                raise ValueError(msg)

    def listall(self):
        """ return the list of all the parameters"""
        ps = [d for d in self.__dict__ if not(d in ['avail', 'doc'])]
        return ps

    def copy(self, obj, list_param):
        """ copy attributes listed in list_param to obj

        On output it returns missing attributes
        """
        missing = []
        for k in list_param:
            if hasattr(self, k):
                setattr(obj, k, getattr(self, k))
            else:
                missing.append(k)
        return missing


if __name__ == "__main__":
    param = Namelist()
#    print('list of parameters')
#    print(param.listall())

    # to have the documentation on one particular parameter
#    param.man('stratification')

    # to get the documentation on all the parameters
    print(param.__dict__)
#    param.checkall()
