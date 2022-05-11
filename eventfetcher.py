#!/usr/bin/env python
import _pickle as cPickle
import logging
import os.path
import re
import sys

from obspy import Stream, read_events
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("EventFetcher")
logger.setLevel(logging.INFO)


def filter_out_channel_without_3channels(waveforms_id, bulk, inventory):
    tmp_bulk = []
    for net, sta, loc, chan, t1, t2 in bulk:
        inv = inventory.select(
            network=net, station=sta, location=loc, time=t1 + (t2 - t1) / 2.0
        )
        # channels = inv[0][0]
        if inv and len(inv[0][0]) == 3:
            tmp_bulk.append((net, sta, loc, chan, t1, t2))
        else:
            w = ".".join((net, sta, loc, chan))
            if inv:
                logger.info(
                    "Filtering out %s (only %d channel(s))" % (w, len(inv[0][0]))
                )
            else:
                logger.info("Filtering out %s (no metadata)" % w)
            id = ".".join((net, sta, loc, chan))
            waveforms_id = cleanup_waveforms_id(waveforms_id, id)
    return waveforms_id, tmp_bulk


def filter_out_station_by_distance(
    waveforms_id, bulk, inventory, latitude, longitude, station_max_dist_km
):
    tmp_bulk = []
    for net, sta, loc, chan, t1, t2 in bulk:
        tmpchan = chan[:-1] + "Z"
        w = ".".join((net, sta, loc, tmpchan))
        t = t1 + (t2 - t1) / 2.0
        try:
            coord = inventory.get_coordinates(w, t)
        except Exception as e:
            logger.error(e)
            continue

        distance, az, baz = gps2dist_azimuth(
            coord["latitude"],
            coord["longitude"],
            latitude,
            longitude,
        )
        # distance in meters, convert it to km
        distance = distance / 1000.0
        if distance <= station_max_dist_km:
            tmp_bulk.append((net, sta, loc, chan, t1, t2))
        else:
            logger.info(
                "Filtering out %s (dist(%.1f) > %.1f)"
                % (w, distance, station_max_dist_km)
            )
            id = ".".join((net, sta, loc, chan))
            waveforms_id = cleanup_waveforms_id(waveforms_id, id)
    return waveforms_id, tmp_bulk


def cleanup_waveforms_id(waveforms_id, id):
    net, sta, loc, chan = id.split(".")
    wid_to_remove = []
    for wid in waveforms_id:
        wid_net, wid_sta, wid_loc, wid_chan = wid.split(".")
        if wid_net == net and wid_sta == sta:
            wid_to_remove.append(wid)

    for wid in wid_to_remove:
        waveforms_id.remove(wid)

    return waveforms_id


def remove_traces_without_3channels(waveforms_id, traces):
    traces_done = []
    traces_to_remove = []
    for i, trace in enumerate(traces):
        stats = trace.stats
        net_sta_loc = ".".join([stats.network, stats.station, stats.location])
        if net_sta_loc in traces_done:
            continue

        tmp = traces.select(
            network=stats.network, station=stats.station, location=stats.location
        )
        if tmp.count() != 3:
            for tr in tmp:
                traces_to_remove.append(tr)
        traces_done.append(net_sta_loc)

    for tr in traces_to_remove:
        net_sta_loc = ".".join(tr.id.split(".")[:3])
        logger.warning(
            "Missing channel for %s: removing trace %s" % (net_sta_loc, tr.id)
        )
        traces.remove(tr)
        cleanup_waveforms_id(waveforms_id, tr.id)

    return waveforms_id


class EventInfo(object):
    """Store basic event information
    - latitude
    - longitude
    - T0
    - qml
    """

    def __init__(self):
        self.id = None
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.T0 = None
        self.event_type = None
        self.qml = None


class EventFetcher(object):
    """Fetch qml and traces for a given event id."""

    def __init__(
        self,
        event_id,
        starttime=None,
        endtime=None,
        starttime_offset=0,
        time_length=60,
        station_max_dist_km=None,
        base_url=None,
        ws_event_url=None,
        ws_station_url=None,
        ws_dataselect_url=None,
        black_listed_waveforms_id=None,
        waveforms_id=None,
        keep_only_3channels_station=False,
        enable_RTrotation=False,
        backup_dirname=".",
        use_cache=False,
        fdsn_debug=False,
    ):

        self.st = None
        self.starttime = starttime
        self.endtime = endtime
        self.starttime_offset = starttime_offset
        self.time_length = time_length
        self.station_max_dist_km = station_max_dist_km
        self.keep_only_3channels_station = keep_only_3channels_station
        self.enable_RTrotation = enable_RTrotation

        # set FDSN clients
        # configuring 3 differents urls doesn't work.
        # we have to split in 2 Fdsn clients
        self.trace_client = Client(
            debug=fdsn_debug,
            base_url=base_url,
            service_mappings={
                "dataselect": ws_dataselect_url,
                "station": ws_station_url,
            },
        )
        self.event_client = Client(
            debug=fdsn_debug, service_mappings={"event": ws_event_url}
        )

        self.backup_event_file = os.path.join(backup_dirname, "{}.qml".format(event_id))
        self.backup_traces_file = os.path.join(
            backup_dirname, "{}.traces".format(event_id)
        )
        if black_listed_waveforms_id:
            self.black_listed_waveforms_id = black_listed_waveforms_id
        else:
            self.black_listed_waveforms_id = []

        self.event = EventInfo()
        self.event.id = event_id
        self.use_cache = use_cache
        self._fetch_data(waveforms_id=waveforms_id)
        self.get_picks()
        self.compute_distance_az_baz()
        if self.enable_RTrotation and self.st:
            st_RT = self.rotate_to_RT()
            self.st += st_RT
        if self.st:
            self.st.sort()
            if logger.level == logging.DEBUG:
                logger.info(self.st.__str__(extended=True))
            else:
                logger.info(self.st)
        else:
            logger.warning("No trace !")

    def _fetch_data(self, waveforms_id=None):
        # Fetch event's traces from ws or cached files
        cat = None
        fetch_from_cache_success = None

        if self.use_cache:
            if os.path.isfile(self.backup_event_file):
                logger.debug(
                    "Fetching event %s from file %s.",
                    self.event.id,
                    self.backup_event_file,
                )
                cat = read_events(self.backup_event_file)
                fetch_from_cache_success = True
            else:
                logger.debug(
                    "Trying to fetch event %s from file. But %s does not exist!",
                    self.event.id,
                    self.backup_event_file,
                )

                fetch_from_cache_success = False

        if not self.use_cache or fetch_from_cache_success is not True:
            logger.info("Fetching event %s from FDSN-WS.", self.event.id)
            cat = self.get_event()

        if not cat:
            logger.error("No event found !")
            return

        try:
            self.event.qml = cat.events[0]
        except Exception as e:
            logger.error(e)
            return

        (
            self.event.latitude,
            self.event.longitude,
            self.event.depth,
        ) = self.get_event_coordinates(self.event.qml)
        self.event.T0 = self.get_event_time(self.event.qml)
        self.event.event_type = self.get_event_type(self.event.qml)

        if waveforms_id:
            self.waveforms_id = waveforms_id
        else:
            self.waveforms_id = self._hack_streams(
                self.get_event_waveforms_id(self.event.qml)
            )
            self.show_pick_offet(self.event.qml)

        # Set time window for trace extraction
        self._set_extraction_time_window()

        # Fetch traces from ws or cached file
        for w in self.black_listed_waveforms_id:
            try:
                self.waveforms_id.remove(w)
            except Exception:
                pass

        fetch_from_cache_success = None
        if self.use_cache:
            if os.path.isfile(self.backup_traces_file):
                logger.info(
                    "Fetching traces from cached file %s.", self.backup_traces_file
                )
                with open(self.backup_traces_file, "rb") as fp:
                    self.st = cPickle.load(
                        fp, fix_imports=True, encoding="ASCII", errors="strict"
                    )

                # remove black listed waveform_id
                for w in self.black_listed_waveforms_id:
                    for tr in self.st.select(id=w):
                        self.st.remove(tr)
                fetch_from_cache_success = True
            else:
                logger.info(
                    "Trying to fetch traces from cached file, but %s does not exist!",
                    self.backup_event_file,
                )
                fetch_from_cache_success = False

        if not self.use_cache or fetch_from_cache_success is not True:
            logger.info("Fetching traces from FDSN-WS.")
            # self.st = self.get_trace(self.starttime, self.endtime)
            self.st = self.get_trace_bulk(self.starttime, self.endtime)

        if self.st == []:
            logger.warning("No traces !")

    def _set_extraction_time_window(self):
        """Set time window for trace extraction"""
        if self.starttime is None:
            self.starttime = self.event.T0
        self.starttime += self.starttime_offset

        if self.endtime is None:
            self.endtime = self.starttime + self.time_length

    def _hack_P_stream(self, waveforms_id):
        waveforms_id = re.sub(r"-$", "?", waveforms_id)

        waveforms_id = re.sub(r"\.HH$", ".HHZ", waveforms_id)
        waveforms_id = re.sub(r"\.EL$", ".ELZ", waveforms_id)
        waveforms_id = re.sub(r"\.HN$", ".HNZ", waveforms_id)

        waveforms_id = re.sub("H.?$", "HZ", waveforms_id)
        waveforms_id = re.sub("L.?$", "LZ", waveforms_id)
        waveforms_id = re.sub("N.?$", "NZ", waveforms_id)
        # waveforms_id = re.sub("N.?$", "NZ", waveforms_id)
        return waveforms_id

    def _hack_streams(self, waveforms_id):
        """Hack to get rid off sc3 users mislabeling phases."""
        waveforms_id = [re.sub("H.?$", "H?", s) for s in waveforms_id]
        waveforms_id = [re.sub("L.?$", "L?", s) for s in waveforms_id]
        waveforms_id = [re.sub("N.?$", "N?", s) for s in waveforms_id]
        # remove multiple same occurence
        waveforms_id = set(waveforms_id)
        return waveforms_id

    def get_trace_bulk(self, starttime, endtime):
        bulk = []
        for w in self.waveforms_id:
            net, sta, loc, chan = w.split(".")
            bulk.append((net, sta, loc, chan, starttime, endtime))

        # get inventory
        try:
            inventory = self.trace_client.get_stations_bulk(bulk, level="response")
        except Exception as e:
            logger.error(e)

        # keep only stations with 3 component
        if self.keep_only_3channels_station:
            self.waveforms_id, bulk = filter_out_channel_without_3channels(
                self.waveforms_id, bulk, inventory
            )

        # get rid off stations too far away
        if self.station_max_dist_km:
            self.waveforms_id, bulk = filter_out_station_by_distance(
                self.waveforms_id,
                bulk,
                inventory,
                self.event.latitude,
                self.event.longitude,
                self.station_max_dist_km,
            )

        # get traces but without response as attach_response does not work as expected
        try:
            traces = self.trace_client.get_waveforms_bulk(bulk, attach_response=False)
        except Exception as e:
            logger.error(e)
            return Stream()

        # merge multiple segments if any
        try:
            traces.merge(method=0, fill_value="interpolate")
        except Exception as e:
            logger.error(e)

        # add inventory to trace
        for i, _w in enumerate(traces):
            _stats = _w.stats
            _wid = ".".join(
                [_stats.network, _stats.station, _stats.location, _stats.channel]
            )
            logger.debug(_wid)
            traces[i].stats.response = inventory.select(
                network=_stats.network,
                station=_stats.station,
                location=_stats.location,
                # channel=_stats.channel,  # all channel have to be included for rotation !!??!!
                time=starttime + (endtime - starttime) / 2.0,
            )
            logger.debug(traces[i].stats.response)

            try:
                traces[i].stats.coordinates = traces[i].stats.response.get_coordinates(
                    _wid
                )
            except Exception as e:
                logger.error("No station coordinates for %s" % (_wid, e))
                traces[i].stats.coordinates = None
            logger.debug("%s: %s", _wid, traces[i].stats.coordinates)

        # Check if 3 channels are present (ie. no missing trace)
        if self.keep_only_3channels_station:
            self.waveforms_id = remove_traces_without_3channels(
                self.waveforms_id, traces
            )

        # Sync all traces to starttime
        traces.trim(starttime=starttime, endtime=endtime)

        # save traces with pickle
        if self.backup_traces_file:
            logger.info("writting to %s", self.backup_traces_file)
            with open(self.backup_traces_file, "wb") as fp:
                cPickle.dump(traces, fp)
        return traces

    def get_trace(self, starttime, endtime):
        """Get waveform using FDSNWS"""
        traces = Stream()
        # print(self.waveforms_id)
        for w in self.waveforms_id:
            logger.debug("Working on %s ... ", w)
            net, sta, loc, chan = w.split(".")

            # get trace
            logger.debug("Start to fetch trace %s [%s-%s]", w, starttime, endtime)
            try:
                waveform = self.trace_client.get_waveforms(
                    net, sta, loc, chan, starttime, endtime, attach_response=False
                )
            except Exception as e:
                logger.error(e)
                continue

            if not waveform:
                logger.debug("No data for trace %s [%s-%s]", w, starttime, endtime)
                continue

            # be sure to have only one segment in trace
            try:
                waveform.merge(method=0, fill_value="interpolate")
            except Exception as e:
                logger.error(e)
                logger.info(waveform)
                continue
            else:
                logger.debug(waveform)

            # get coordinates since attach_response seems not to be enough
            logger.debug("Start to fetch inventory for %s", w)
            try:
                inventory = self.trace_client.get_stations(
                    network=net,
                    station=sta,
                    location=loc,
                    channel=chan,
                    starttime=starttime,
                    endtime=endtime,
                    level="response",
                )
            except Exception as e:
                logger.error(e)
                continue

            logger.debug(inventory)

            for i, _w in enumerate(waveform):
                _stats = _w.stats
                _wid = ".".join(
                    [_stats.network, _stats.station, _stats.location, _stats.channel]
                )
                logger.debug(_wid)
                waveform[i].stats.response = inventory
                try:
                    waveform[i].stats.coordinates = inventory.get_coordinates(_wid)
                except Exception as e:
                    logger.error(e)
                    waveform[i].stats.coordinates = None
                logger.debug("%s: %s", _wid, waveform[i].stats.coordinates)

            # store trace
            traces += waveform

        # Sync all traces to starttime
        traces.trim(starttime=starttime, endtime=endtime)

        # save traces with pickle
        if self.backup_traces_file:
            logger.info("writting to %s", self.backup_traces_file)
            with open(self.backup_traces_file, "wb") as fp:
                cPickle.dump(traces, fp)
        return traces

    def rotate_to_RT(self):
        # make a copy and rotate traces
        # return only R and T traces
        if not hasattr(self, "waveforms_id"):
            return

        wids = []
        for w in self.waveforms_id:
            logger.debug("Working on %s ... ", w)
            net, sta, loc, chan = w.split(".")
            wids.append(".".join((net, sta, loc, "*")))
        wids = set(wids)

        # print(wids)
        st_RT = Stream()
        stcopy = self.st.copy()

        for wid in wids:
            st = stcopy.select(id=wid)
            st._trim_common_channels()

            try:
                logger.debug("Rotating %s" % wid)
                inventory = st[0].stats.response  # All channel should be included here
                # nb_channel = len(inventory.get_contents()["channels"])
                # if nb_channel != 3:
                #    logger.warning("%s has only %d channel" % (wid, nb_channel))
                #    raise
                st.rotate(method="->ZNE", inventory=inventory)
                st.rotate(method="NE->RT", inventory=inventory)
            except Exception as e:
                logger.warning("Can't rotate: %s (%s)" % (wid, e))
                logger.warning(st)
            else:
                # logger.warning(st)
                # remove Z trace
                for tr in st.select(component="Z"):
                    st.remove(tr)
                st_RT += st

        return st_RT

    def get_event(self):
        try:
            cat = self.event_client.get_events(
                eventid=self.event.id, includearrivals=True
            )
        except Exception as e:
            logger.error("Error getting event = %s" % self.event.id)
            logger.error(e)
            sys.exit()

        if self.backup_event_file:
            logger.info(
                "writting event (%s) to quakeml file %s",
                self.event.id,
                self.backup_event_file,
            )
            cat.write(self.backup_event_file, format="QUAKEML")
        return cat

    def get_event_coordinates(self, e):
        o = e.preferred_origin()
        return o.latitude, o.longitude, o.depth / 1000.0

    def get_event_time(self, e):
        o = e.preferred_origin()
        return o.time

    def get_event_type(self, e):
        return e.event_type

    def get_event_waveforms_id(self, e):
        waveforms_id = []
        o = e.preferred_origin()
        for a in o.arrivals:
            for p in e.picks:
                if a.pick_id == p.resource_id:
                    wfid = self._hack_P_stream(p.waveform_id.get_seed_string())
                    waveforms_id.append(wfid)
                    # waveforms_id.append(p.waveform_id.get_seed_string())
                    break
        return waveforms_id

    def compute_distance_az_baz(self):
        # Calculating distance and azimuth from station to event
        if not self.st:
            return
        for tr in self.st:
            if "coordinates" not in tr.stats or tr.stats.coordinates is None:
                logger.warning("compute_distance_az_baz: no coordinates for %s" % tr)
                continue

            distance, az, baz = gps2dist_azimuth(
                tr.stats.coordinates.latitude,
                tr.stats.coordinates.longitude,
                self.event.latitude,
                self.event.longitude,
            )
            tr.stats.distance = distance  # in meters
            tr.stats.back_azimuth = az

    def get_picks(self, e=None):
        self.picks = {}
        if e is None:
            e = self.event.qml
            if e is None:
                return

        o = e.preferred_origin()
        t0 = o.time
        for a in o.arrivals:
            if not a.phase.startswith("P"):
                continue
            for p in e.picks:
                if a.pick_id == p.resource_id:
                    wfid = self._hack_P_stream(p.waveform_id.get_seed_string())
                    self.picks[wfid] = {
                        "time": p.time,
                        "offset": p.time - (t0 + self.starttime_offset),
                    }
                    break

    def show_pick_offet(self, e=None):
        if e is None:
            e = self.event.qml

        o = e.preferred_origin()
        t0 = o.time
        for a in o.arrivals:
            if not a.phase.startswith("P"):
                continue
            for p in e.picks:
                if a.pick_id == p.resource_id:
                    logger.debug(
                        "%s %s %s",
                        self._hack_P_stream(p.waveform_id.get_seed_string()),
                        p.time,
                        p.time - t0,
                    )
                    break


def _test():
    # ReNaSS
    ws_base_url = "http://10.0.1.36"
    ws_event_url = "http://10.0.1.36:8080/fdsnws/event/1/"
    ws_station_url = "http://10.0.1.36:8080/fdsnws/station/1/"
    ws_dataselect_url = "http://10.0.1.36:8080/fdsnws/dataselect/1/"

    # event
    # event_id = 'eost2019uhsagsbu'
    # event_id = 'eost2020vvqguwny'
    event_id = "eost2021nvpzrzto"

    # get data
    mydata = EventFetcher(
        event_id,
        base_url=ws_base_url,
        ws_event_url=ws_event_url,
        ws_station_url=ws_station_url,
        ws_dataselect_url=ws_dataselect_url,
        use_cache=False,
    )

    if not mydata.st:
        logger.info("No data associated to event %s", event_id)
        sys.exit()


if __name__ == "__main__":
    _test()
