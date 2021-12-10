#!/usr/bin/python

# Reads two spectra and calculates various R-factors -- FS 2011

import numpy
from scipy import interpolate
import sys
from optparse import OptionParser


def error(msg):
   """ write error message and quit
   """
   sys.stderr.write(msg + "\n")
   sys.exit(3)


def deriv(spec,h):
   """ calculate first derivative of function 'spec'
       using the central finite difference method up to 6th order,
       and for the first 3 and last 3 grid points the
       forward/backward finite difference method up to 2nd order.
       ...as used in f77-program and suggested by Zanazzi-Jona...
   """ 
   der_spec =[[i[0],0] for i in spec]

   length=len(spec)
   for i in range(3,length-3):
      der_spec[i][1]=(-1*spec[i-3][1]+9*spec[i-2][1]-45*spec[i-1][1]+45*spec[i+1][1]-9*spec[i+2][1]+1*spec[i+3][1])/(60*h)
   for i in range(0,3):
      der_spec[i][1]=(-11*spec[i][1]+18*spec[i+1][1]-9*spec[i+2][1]+2*spec[i+3][1])/(6*h)
   for i in range(length-3,length):
      der_spec[i][1]=(11*spec[i][1]-18*spec[i-1][1]+9*spec[i-2][1]-2*spec[i-3][1])/(6*h)

   return der_spec


def get_range(tspec,espec,w_incr,shift,start,stop):
   """ determine wavenumber range within the comparison between theoretical
       and experimental spectrum is performed (depends on the shift)
   """
   de1=start+shift-espec[0][0]
   if (de1 >= 0 ):
      de1=int((start+shift-espec[0][0])/w_incr+0.00001)
      enstart=de1
      tnstart=int((start-tspec[0][0])/w_incr+0.00001)
   else:
      de1=int((start+shift-espec[0][0])/w_incr-0.00001)
      enstart=0
      tnstart=int((start-tspec[0][0])/w_incr-de1+0.00001)
   de2=stop+shift-espec[-1][0]
   if (de2 <= 0 ):
      de2=int((stop+shift-espec[-1][0])/w_incr-0.00001)
      enstop=len(espec)+de2
      tnstop=len(tspec)+int((stop-tspec[-1][0])/w_incr-0.00001) 
   else:
      de2=int((stop+shift-espec[-1][0])/w_incr+0.00001)
      enstop=len(espec)
      tnstop=len(tspec)+int((stop-tspec[-1][0])/w_incr-de2-0.00001)
   return tnstart, tnstop, enstart, enstop
 

def integrate(integrand,delta):
   """ integrate using the trapezoid method as Zanazzi-Jona suggested and was used in the f77-program...
   """
   integral = 0.5*(integrand[0][1]+integrand[-1][1])   
   for i in range(1,len(integrand)-1):
      integral += integrand[i][1]
   return integral*delta

def ypendry(spec,d1_spec,VI):
   """ calculate the Pendry Y-function: y=l^-1/(l^-2+VI^2) with l=I'/I (logarithmic derivative),
       J.B. Pendry, J. Phys. C: Solid St. Phys. 13 (1980) 937-44
   """
   y=[[i[0],0] for i in spec]

   for i in range(len(spec)):
      if (abs(spec[i][1]) <= 1.E-7):
         if (abs(d1_spec[i][1]) <= 1.E-7):
            y[i][1] = 0 
         else:
            y[i][1] = (spec[i][1]/d1_spec[i][1])/((spec[i][1]/d1_spec[i][1])**2+VI**2)
      else:
         y[i][1] = (d1_spec[i][1]/spec[i][1])/(1+(d1_spec[i][1]/spec[i][1])**2*(VI**2))
   return y

def _main():

   usage = """ %prog [options] r-fac.in
        Reads two spectra and calculates various R-factors -- FS 2011

        Attention: both spectra have to be given on the same, equidistant grid!
        NOTE: in the f77-program R1 is scaled by 0.75 and R2 is scaled by 0.5; this is not done here
        Please provide a file r-fac.in with the following specifications (without the comment lines!!!) 
        (the numbers are just examples, choose them according to your particular case)
        start=1000       # where to start the comparison
        stop=1800        # where to stop the comparison
        w_incr=0.5       # grid interval of the spectra -- should be 1 or smaller! (otherwise integrations/derivatives are not accurate)
        shift_min=-10    # minimal shift of the theoretical spectrum 
        shift_max=+10    # maximal shift of the experimental spectrum
        shift_incr=1     # shift interval
        r=pendry         # which r-factor should be calculated? options: pendry, ZJ, R1, R2 (give a list of the requested r-factors separated by comma)
        VI=10            # approximate half-width of the peaks (needed for pendry r-factor)
        """
   parser = OptionParser(usage=usage)
   parser.add_option("-t","--theory", 
                     default="theory.dat",dest="theory",metavar="filename",
                     help="input file theoretical spectrum [default: %default]")
   parser.add_option("-e","--exp", 
                     default="exp.dat",dest="exp",metavar="filename",
                     help="input file experimental spectrum [default: %default]")

   options, args = parser.parse_args()
   
# read spectra in
   input_theory=open(options.theory)
   tspec=[list(map(float,line.split())) for line in input_theory.readlines()]
   input_theory.close()
   input_exp=open(options.exp)
   espec=[list(map(float,line.split())) for line in input_exp.readlines()]
   input_exp.close()

# read r-fac.in
   if (len(args) != 1):
      error("Please provide r-fac.in and execute: python calc-r-factors.py r-fac.in")
   for line in open(args[0]):
      line=line.split("=")
      if line[0]=="start":
         start=float(line[1])
      elif line[0]=="stop":
         stop=float(line[1])
      elif line[0]=="w_incr":
         w_incr=float(line[1])
      elif line[0]=="shift_min":
         shift_min=float(line[1])
      elif line[0]=="shift_max":
         shift_max=float(line[1])
      elif line[0]=="shift_incr":
         shift_incr=float(line[1])
      elif line[0]=="r":
         r=line[1].split(",") 
         r=[i.strip() for i in r]  # strip off leading and trailing whitespaces
         sys.stdout.write("Requested r-factors: ")
         for i in r:
            sys.stdout.write("%s " % i)
         sys.stdout.write("\nNOTE: in the f77-program R1 is scaled by 0.75 and R2 is scaled by 0.5; this is not done here\n\n")   
      elif line[0]=="VI":
         VI=float(line[1])

# perform some checks of the input data...
   if (int(shift_incr/w_incr+0.00001) == 0):
      error("Error: shift_incr cannot be smaller than w_incr!")
   if (start-espec[0][0] < 0) or (espec[-1][0]-stop < 0):
      error("check experimental spectrum!!")
   if (start-tspec[0][0] < 0) or (tspec[-1][0]-stop < 0):
      error("check theoretical spectrum!!")
   if (int((espec[-1][0]-espec[0][0])/w_incr+0.0001) != len(espec)-1 ) or (int((tspec[-1][0]-tspec[0][0])/w_incr+0.0001) != len(tspec)-1 ):
      error("check w_incr!!")

 
# cut out data points that are not needed in order to save time...
   if (espec[0][0]-(start+shift_min-w_incr*25) < 0):
         espec=espec[-1*int((espec[0][0]-(start+shift_min-w_incr*25))/w_incr-0.00001):][:]
   if (espec[-1][0]-(stop+shift_max+w_incr*25) > 0):
         espec=espec[:-1*(int((espec[-1][0]-(stop+shift_max+w_incr*25))/w_incr+0.00001)+1)][:] 
   if (tspec[0][0]-(start-w_incr*25) < 0):
         tspec=tspec[-1*int((tspec[0][0]-(start-w_incr*25))/w_incr-0.00001):][:]
   if (tspec[-1][0]-(stop+w_incr*25) > 0):
         tspec=tspec[:-1*(int((tspec[-1][0]-(stop+w_incr*25))/w_incr+0.00001)+1)][:]

   
# set negative intensity values to zero
   for i in range(0,len(espec)):
      if (espec[i][1]<0):
         espec[i][1]=0
   for i in range(0,len(tspec)):
      if (tspec[i][1]<0):
         tspec[i][1]=0
   
# start calculating derivatives...
   d1_espec = deriv(espec,w_incr)   
   d1_tspec = deriv(tspec,w_incr)
# calculate the second derivatives if the Zanazzi-Jona R-factor is requested   
   if "ZJ" in r:
      d2_tspec = deriv(d1_tspec,w_incr)
      d2_espec = deriv(d1_espec,w_incr)
# calculate Pendry Y-function if Pendry R-factor is requested      
   if "pendry" in r:
      ye = ypendry(espec,d1_espec,VI)
      yt = ypendry(tspec,d1_tspec,VI)
   


   min_pendry = [1.E100,0]
   min_r1     = [1.E100,0]
   min_r2     = [1.E100,0]
   min_zj     = [1.E100,0]
# start with loop over x-axis shifts
   for shift in numpy.arange(shift_min,shift_max+shift_incr,shift_incr):
      # get the interval within the two spectra are compared
      tnstart,tnstop,enstart,enstop = get_range(tspec,espec,w_incr,shift,start,stop) 
      sys.stdout.write("\nshift: %9.3f, theory-start: %5d, theory-end: %5d, exp-start: %5d, exp-end: %5d\n" % (shift,tspec[tnstart][0],tspec[tnstop-1][0],espec[enstart][0],espec[enstop-1][0]))
      s_espec = numpy.array(espec[enstart:enstop]) # cut out the interval within which the comparison takes place
      s_tspec = numpy.array(tspec[tnstart:tnstop])
      s_d1_espec = numpy.array(d1_espec[enstart:enstop])
      s_d1_tspec = numpy.array(d1_tspec[tnstart:tnstop])
      c_scale=integrate(s_espec,w_incr)/integrate(s_tspec,w_incr)
      if "pendry" in r:
         # see J.B. Pendry, J. Phys. C: Solid St. Phys. 13 (1980) 937-44
         s_yt = numpy.array(yt[tnstart:tnstop]) # cut out the interval within which the comparison takes place
         s_ye = numpy.array(ye[enstart:enstop])
         te2 = integrate((s_yt-s_ye)**2,w_incr) # integrate (yt-ye)^2
         t2e2 = integrate(s_yt**2+s_ye**2,w_incr) # integrate yt^2+ye^2
         r_pend = te2/t2e2
         sys.stdout.write("Pendry R-factor : %f, shift: %f\n" % (r_pend,shift))
         if (r_pend < min_pendry[0] ):
            min_pendry=[r_pend,shift]
      if "R1" in r:
         # see  M.A. van Hove, S.Y. Tong, and M.H. Elconin, Surfac Science 64 (1977) 85-95
         r1 = integrate(abs(s_espec-c_scale*s_tspec),w_incr)/integrate(abs(s_espec),w_incr)
         sys.stdout.write("R1 R-factor     : %f, shift: %f\n" % (r1,shift))
         if (r1 < min_r1[0]):
            min_r1=[r1,shift]
      if "R2" in r:
         # see  M.A. van Hove, S.Y. Tong, and M.H. Elconin, Surfac Science 64 (1977) 85-95
         r2 = integrate((s_espec-c_scale*s_tspec)**2,w_incr)/integrate(s_espec**2,w_incr)
         sys.stdout.write("R2 R-factor     : %f, shift: %f\n" % (r2,shift))
         if (r2 < min_r2[0]):
            min_r2=[r2,shift]
      if "ZJ" in r:      
         # E. Zanazzi, F. Jona, Surface Science 62 (1977), 61-88
         s_d2_tspec = numpy.array(d2_tspec[tnstart:tnstop])
         s_d2_espec = numpy.array(d2_espec[enstart:enstop])

         epsilon = 0
         for i in s_d1_espec:
            if abs(i[1]) > epsilon:
               epsilon = abs(i[1])
         
         integrand = abs(c_scale*s_d2_tspec-s_d2_espec)*abs(c_scale*s_d1_tspec-s_d1_espec)/(abs(s_d1_espec)+epsilon)
         # interpolate integrand onto 10 times denser grid, see publication by Zanazzi & Jona
         incr = 0.1*w_incr
         grid_old = numpy.arange(0,len(integrand))*w_incr
         grid_new = numpy.arange(grid_old[0],grid_old[-1]+incr,incr)
         spl = interpolate.splrep(grid_old,integrand.T[1],k=3,s=0)
         integrand_dense = interpolate.splev(grid_new,spl,der=0)
         integrand_dense = numpy.vstack((grid_new,integrand_dense)).T
         # calculate reduced Zanazzi-Jona R-factor r=r/0.027
         r_zj = integrate(integrand_dense,incr)/(0.027*integrate(abs(s_espec),w_incr))
         sys.stdout.write("red. ZJ R-factor: %f, shift %f\n" % (r_zj,shift))
         if (r_zj < min_zj[0]):
            min_zj=[r_zj,shift]


# find minimal r-factor and write it out
   sys.stdout.write("\nMinimal r-factors:\n")
   if "pendry" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, Pendry R-factor = %7.5f \n" % ( min_pendry[1], min_pendry[0]))
   if "R1" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, R1 R-factor = %7.5f \n" % ( min_r1[1], min_r1[0]))
   if "R2" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, R2 R-factor = %7.5f \n" % ( min_r2[1], min_r2[0]))
   if "ZJ" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, ZJ R-factor = %7.5f \n" % ( min_zj[1], min_zj[0]))


if __name__ == "__main__":
      _main()
